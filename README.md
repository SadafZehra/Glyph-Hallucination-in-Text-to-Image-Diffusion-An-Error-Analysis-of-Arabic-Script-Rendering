# Glyph Hallucination in Text-to-Image Diffusion: An Error Analysis of Arabic Script Rendering

**Notebook:** `Arabic_Glyph_Hallucination.ipynb` (Google Colab, T4 GPU, about 60 to 90 minutes)
**Base model:** Stable Diffusion 1.5 · **OCR:** EasyOCR (Arabic) · **Adaptation:** LoRA

---

## 1. Research question

When Stable Diffusion is asked to draw an Arabic word, how often does it write the correct letters, and what kinds of mistakes does it make?

## 2. Pipeline

1. **Word list.** Built from the Kaggle corpus *Arabic News Articles from Aljazeera.net* (`arhouati/arabic-news-articles-from-aljazeeranet`, downloaded with `kagglehub`, no login). Stop words are removed. A built-in word list is used as a fallback if the download fails.
2. **Clean rendering and OCR verification.** Each word is rendered with 8 open-licence Arabic fonts (Amiri, Scheherazade, Lateef, Tajawal, Tajawal-Bold, Harmattan, Mada, Cairo) using `arabic-reshaper` and `python-bidi`. A word is kept only if EasyOCR reads it exactly in at least 2 of 3 random fonts. This way later errors can be blamed on the diffusion model and not on the OCR.
3. **Baseline generation.** Stable Diffusion 1.5 generates each unseen test word with two prompts (a wooden sign and a poster), 25 steps, guidance 7.5, 512 px. Images are read back with OCR.
4. **Error analysis**, with bootstrap confidence intervals:
   - character error rate (CER) and exact match
   - images with no Arabic text, and Arabic-script ratio
   - substitutions, deletions and insertions per letter
   - **dot errors**: substitutions between letters that share a skeleton (for example س and ش)
   - **error rate by letter position**: isolated, initial, medial, final
   - CLIP score (image quality, computed without the word)
5. **LoRA fine-tuning** (rank 8, 1500 steps, learning rate 1e-4, U-Net and CLIP text encoder) on rendered Arabic words with varied fonts, colours, sizes, rotation and blur. Training words never overlap with test words.
6. **Paired evaluation** of baseline vs. LoRA on the same words, prompts and seeds, with a paired bootstrap on the CER difference.

## 3. Setup actually used in the saved run

| Item | Value |
|---|---|
| Candidate words from corpus | 17,456 (17,383 after filtering) |
| Words checked / OCR-verified | 920 checked, 177 verified |
| Test words | 60 (120 images with 2 prompts) |
| Training words | 117 (the settings asked for 400, but only 177 words passed verification) |
| OCR on clean renders (Amiri) | CER 0.005 |

## 4. Results

**Baseline Stable Diffusion 1.5 (120 images):**

| Metric | Value |
|---|---|
| CER (mean) | **0.954** (95 % CI 0.937 to 0.970) |
| Exact match | **0 %** |
| Images with no Arabic text | 18.3 % |
| Arabic-script ratio | 0.703 |
| Substitutions per letter | 0.460 |
| Deletions per letter | 0.319 |
| Insertions per letter | 1.830 |
| Dot errors (% of substitutions) | 5.6 % |
| CLIP score | 0.343 |

**Error rate by letter position:**

| Isolated | Initial | Medial | Final |
|---|---|---|---|
| 0.500 | 0.758 | 0.882 | 0.803 |

Reading: the model essentially never writes a correct Arabic word. It tends to *invent extra letters* (insertions are by far the largest error type), produces Arabic-looking strings that are not the requested word, and is worst on medial letters, which are the connected forms. Isolated letters, the simplest shapes, are the least wrong. Only a small share of substitutions are pure dot confusions, so most mistakes are not dot-level slips.








4. Separate analysis of ligatures and diacritics.

## 10. Requirements

`kagglehub`, `diffusers`, `transformers`, `accelerate`, `peft`, `easyocr`, `arabic-reshaper`, `python-bidi`, PyTorch, pandas, matplotlib. The notebook installs them in its first cell.
