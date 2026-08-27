# Information

- In this directory, `attention_mask` uses **1 for tokens to keep** and **0 for tokens to ignore (masked)**.  
  (This applies both to the `smiles` attention mask and the `data` attention mask.)  
  When inputting into PyTorch’s `transformer_encoder`, tokens to ignore must be set to **True**, and tokens to keep must be set to **False**. (Needs confirmation.)  

- Since CLS was not added at the beginning before, pooling was done by averaging positions where attention = 1.  
  (Now that CLS is added at the beginning, this needs to be changed.)  

- Although the name is `freq_attention_masks`, the structure is the same for **freq, IR, and Raman**, so it can be used for IR and Raman as well.  

- For heavy atoms ≥ 26, no train/valid/test splits or directory separation are performed.  
  Currently, they are only used for evaluation.  

---

### Number of data points

| Heavy atom count | Dataset size |
| ---- | ---- |
| 26 | 4312 (≤225 atoms) |
| 36 | 768 (≥226 atoms are 84 items) |

---

### SMILES length

- `tokenizer.tokenize` (used for length validation) does **not** include CLS or EOS.  
- `call` (used for data creation) automatically adds CLS and EOS.  

| Number of heavy atoms | Max length |
| ---- | ---- |
| 5–25 | 26 (excluding CLS, EOS) |
| Value used in model | 32 (including CLS, EOS) |

---

- The `max_spectrum_length` for GDB9 is set to 100.  
  The actual maximum was 81, but 100 is used to allow for extrapolation.  

- In MMP05percent with heavy atom = 12, the length exceeded 100 for the first time—so that’s the approximate scale.  
