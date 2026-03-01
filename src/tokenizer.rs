use std::collections::HashMap;
use anyhow::{Result, bail};
use crate::gguf::{GgufContext, GgufValue};

pub struct Tokenizer {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), String>,
}

impl Tokenizer {
    pub fn from_gguf(ctx: &GgufContext) -> Result<Self> {
        let tokens = match ctx.metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(arr)) => {
                let mut res = Vec::new();
                for val in arr {
                    if let GgufValue::String(s) = val {
                        res.push(s.clone());
                    } else {
                        bail!("Non-string in tokenizer.ggml.tokens");
                    }
                }
                res
            }
            _ => bail!("Missing tokenizer.ggml.tokens"),
        };

        let mut token_to_id = HashMap::new();
        for (i, token) in tokens.iter().enumerate() {
            token_to_id.insert(token.clone(), i as u32);
        }

        let merges = match ctx.metadata.get("tokenizer.ggml.merges") {
            Some(GgufValue::Array(arr)) => {
                let mut res = HashMap::new();
                for val in arr {
                    if let GgufValue::String(s) = val {
                        let parts: Vec<&str> = s.split(' ').collect();
                        if parts.len() == 2 {
                            let combined = format!("{}{}", parts[0], parts[1]);
                            res.insert((parts[0].to_string(), parts[1].to_string()), combined);
                        }
                    }
                }
                res
            }
            _ => HashMap::new(), // Some models might not have merges in the standard way
        };

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            merges,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let special_tokens = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|rep_2|>", "<|rep_3|>"
        ];
        
        let mut tokens = Vec::new();
        let mut remaining = text;
        
        while !remaining.is_empty() {
            let mut found_special = false;
            for &st in &special_tokens {
                if remaining.starts_with(st) {
                    if let Some(&id) = self.token_to_id.get(st) {
                        tokens.push(id);
                        remaining = &remaining[st.len()..];
                        found_special = true;
                        break;
                    }
                }
            }
            
            if found_special {
                continue;
            }
            
            // Find next special token index
            let mut next_special_idx = remaining.len();
            for &st in &special_tokens {
                if let Some(idx) = remaining.find(st) {
                    if idx < next_special_idx {
                        next_special_idx = idx;
                    }
                }
            }
            
            let current_chunk = &remaining[..next_special_idx];
            remaining = &remaining[next_special_idx..];
            
            // Encode regular text chunk
            if !current_chunk.is_empty() {
                let mapped_text = current_chunk.replace(" ", "Ġ").replace("\n", "Ċ");
                let mut words: Vec<String> = mapped_text.chars().map(|c| c.to_string()).collect();
                
                // Iteratively merge
                loop {
                    let mut best_pair = None;
                    for i in 0..words.len().saturating_sub(1) {
                        let pair = (words[i].clone(), words[i+1].clone());
                        if self.merges.contains_key(&pair) {
                            best_pair = Some((i, pair));
                            break; 
                        }
                    }

                    if let Some((i, pair)) = best_pair {
                        let merged = self.merges.get(&pair).unwrap().clone();
                        words.remove(i);
                        words[i] = merged;
                    } else {
                        break;
                    }
                }

                for word in words {
                    if let Some(&id) = self.token_to_id.get(&word) {
                        tokens.push(id);
                    } else {
                        // Fallback
                    }
                }
            }
        }

        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut res = String::new();
        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                // Handle special characters like GGUF's space representation (often \u{2581} for SentencePiece, or Ġ/Ċ for Tiktoken)
                let clean_token = token
                    .replace("\u{2581}", " ")
                    .replace("Ġ", " ")
                    .replace("Ċ", "\n");
                res.push_str(&clean_token);
            }
        }
        res
    }
}
