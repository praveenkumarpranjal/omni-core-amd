use anyhow::Result;
use std::time::Instant;
use rand::Rng;

pub mod gguf;
pub mod hip;
pub mod graph;
pub mod tokenizer;

fn argmax(logits: &[f32]) -> u32 {
    let mut max_idx = 0u32;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i as u32;
        }
    }
    max_idx
}

fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> u32 {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    
    // apply temperature
    let mut probs: Vec<(usize, f32)> = logits.iter()
        .enumerate()
        .map(|(i, &l)| (i, l / temperature))
        .collect();
        
    // sort descending
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // top-k
    let k = top_k.min(probs.len());
    probs.truncate(k);
    
    // softmax
    let max_logit = probs[0].1;
    let mut exp_probs: Vec<(usize, f32)> = probs.into_iter()
        .map(|(i, l)| (i, (l - max_logit).exp()))
        .collect();
        
    let sum: f32 = exp_probs.iter().map(|(_, p)| p).sum();
    for (_, p) in &mut exp_probs {
        *p /= sum;
    }
    
    // top-p
    let mut cumsum = 0.0;
    let mut p_idx = exp_probs.len();
    for (idx, &(_, p)) in exp_probs.iter().enumerate() {
        cumsum += p;
        if cumsum > top_p {
            p_idx = idx + 1;
            break;
        }
    }
    if p_idx > 0 {
        exp_probs.truncate(p_idx);
    }
    
    // re-normalize after top-p
    let sum2: f32 = exp_probs.iter().map(|(_, p)| p).sum();
    
    // sample
    let mut rng = rand::thread_rng();
    let r: f32 = rng.gen::<f32>() * sum2;
    let mut acc = 0.0;
    for &(i, p) in &exp_probs {
        acc += p;
        if r <= acc {
            return i as u32;
        }
    }
    
    exp_probs.last().unwrap().0 as u32
}

fn is_eos(token_id: u32, eos_tokens: &[u32]) -> bool {
    eos_tokens.contains(&token_id)
}

const COLOR_RESET: &str = "\x1b[0m";
const COLOR_BOLD: &str = "\x1b[1m";
const COLOR_CYAN: &str = "\x1b[36m";
const COLOR_GRAY: &str = "\x1b[90m";
const COLOR_PURPLE: &str = "\x1b[35m";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = "../llama.cpp/models/smollm2-1.7b-instruct-q4_k_m.gguf".to_string();
    let mut prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n".to_string();
    let mut max_tokens: usize = 2048; // Higher default
    let mut interactive = false;
    let mut temperature: f32 = 0.0;
    let mut top_k: usize = 40;
    let mut top_p: f32 = 0.95;

    let mut i = 1;
    while i < args.len() {
        if args[i] == "-m" && i + 1 < args.len() {
            model_path = args[i+1].clone();
            i += 2;
        } else if args[i] == "-p" && i + 1 < args.len() {
            prompt = args[i+1].clone();
            i += 2;
        } else if args[i] == "-n" && i + 1 < args.len() {
            max_tokens = args[i+1].parse().unwrap_or(2048);
            if max_tokens == 0 { max_tokens = usize::MAX; } // Uncapped
            i += 2;
        } else if args[i] == "-t" && i + 1 < args.len() {
            temperature = args[i+1].parse().unwrap_or(0.0);
            i += 2;
        } else if args[i] == "--top-k" && i + 1 < args.len() {
            top_k = args[i+1].parse().unwrap_or(40);
            i += 2;
        } else if args[i] == "--top-p" && i + 1 < args.len() {
            top_p = args[i+1].parse().unwrap_or(0.95);
            i += 2;
        } else if args[i] == "-i" || args[i] == "--interactive" {
            interactive = true;
            i += 1;
        } else {
            // fallback (old style)
            if i == 1 { model_path = args[i].clone(); }
            if i == 2 { prompt = args[i].clone(); }
            i += 1;
        }
    }

    // --- Silent Initialization ---
    hip::init_gpu().ok(); 
    let mut graph = graph::LlamaGraph::new()?;
    let ctx = gguf::GgufContext::load(&model_path)?;
    graph.load_weights(&ctx)?;
    let tok = tokenizer::Tokenizer::from_gguf(&ctx)?;

    // Warm-up triggers any internal library logging (like ggml_cuda_init)
    let _ = graph.forward(0, 0); 
    
    // --- Elegant UI Display ---
    // Clear screen and reset cursor to top left to hide all previous init logs
    print!("\x1b[2J\x1b[H"); 
    println!("{}{}● Omni-Core AMD GPU Engine{}", COLOR_BOLD, COLOR_CYAN, COLOR_RESET);
    println!();

    if !interactive {
        println!("{}Prompt:{} {}", COLOR_BOLD, COLOR_RESET, prompt);
    }

    print!("{}{}Assistant:{} ", COLOR_BOLD, COLOR_PURPLE, COLOR_RESET);
    std::io::Write::flush(&mut std::io::stdout()).ok();
    std::io::Write::flush(&mut std::io::stdout()).ok();

    let eos_ids: Vec<u32> = match ctx.metadata.get("tokenizer.ggml.eos_token_id") {
        Some(gguf::GgufValue::Uint32(v)) => vec![*v],
        _ => vec![2, 151643, 151645], 
    };

    let mut all_tokens: Vec<u32> = tok.encode(&prompt);
    let mut pos = 0;
    let mut total_gen_tokens = 0;
    let mut start_gen = Instant::now();

    loop {
        // OPTIMIZATION: Process initial prompt tokens in batch for faster prefill
        let prefill_start = Instant::now();
        let last_logits = if pos < all_tokens.len() {
            let batch_tokens = &all_tokens[pos..];
            let logits = graph.forward_batch(batch_tokens, pos)?;
            pos = all_tokens.len();
            logits
        } else {
            Vec::new()
        };
        let prefill_time = prefill_start.elapsed().as_secs_f64();
        
        if !last_logits.is_empty() && prefill_time > 0.01 {
            // Show prefill performance for initial prompt
            let prefill_tokens = all_tokens.len();
            eprintln!("{}[Prefill: {} tokens in {:.3}s = {:.1} tok/s]{}", 
                COLOR_GRAY, prefill_tokens, prefill_time, 
                prefill_tokens as f64 / prefill_time, COLOR_RESET);
        }
        
        start_gen = Instant::now();
        total_gen_tokens = 0;
        let mut last_logits = last_logits;

        for _ in 0..max_tokens {
            let next_token = sample(&last_logits, temperature, top_k, top_p);
            if is_eos(next_token, &eos_ids) {
                break;
            }

            all_tokens.push(next_token);
            print!("{}", tok.decode(&[next_token]));
            std::io::Write::flush(&mut std::io::stdout()).ok();

            last_logits = graph.forward(next_token, pos)?;
            pos += 1;
            total_gen_tokens += 1;
        }

        let elapsed = start_gen.elapsed().as_secs_f64();
        if total_gen_tokens > 0 {
            println!("\n{}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{}", COLOR_GRAY, COLOR_RESET);
            println!("{}Performance:{} {:.2} tokens/sec ({} tokens in {:.2}s)", 
                COLOR_BOLD, COLOR_RESET, 
                total_gen_tokens as f64 / elapsed,
                total_gen_tokens, elapsed);
        }

        if interactive {
            println!("\n");
            print!("{}{}User:{} ", COLOR_BOLD, COLOR_CYAN, COLOR_RESET);
            std::io::Write::flush(&mut std::io::stdout()).ok();
            
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line).is_err() {
                break;
            }
            let line = line.trim();
            if line.eq_ignore_ascii_case("/exit") || line.eq_ignore_ascii_case("exit") {
                break;
            }
            if line.is_empty() {
                continue;
            }
            
            let user_msg = format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", line);
            let new_tokens = tok.encode(&user_msg);
            all_tokens.extend(&new_tokens);
            
            println!();
            print!("{}{}Assistant:{} ", COLOR_BOLD, COLOR_PURPLE, COLOR_RESET);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        } else {
            println!();
            break;
        }
    }

    Ok(())
}
