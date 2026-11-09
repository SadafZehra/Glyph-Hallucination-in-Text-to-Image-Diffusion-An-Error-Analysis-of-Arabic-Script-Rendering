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

**LoRA fine-tuning.** In the saved run the LoRA evaluation gave **exactly the same numbers as the baseline** in every metric (CER 0.954, CER change +0.000 with CI +0.000 to +0.000). Training loss looked normal (about 0.010 to 0.013), but identical outputs to three decimals mean the LoRA weights almost certainly did not affect the pipeline used for evaluation. **This is a bug or setup issue to fix, and the result should not be reported as "LoRA does not help".** See section 6.

## 5. Limitations

- One base model (SD 1.5), one OCR engine, short single words, two prompt templates.
- Only 117 training words because few words passed OCR verification; stricter verification also biases the set toward common, easy words.
- OCR on generated images is itself noisy, so CER is a rough measure. The very high insertion rate partly reflects OCR reading background texture as letters.
- The LoRA comparison is currently invalid (see below).

## 6. What to fix before reporting the LoRA part

1. Check that the LoRA-adapted U-Net and text encoder are the modules actually used by `pipe` during evaluation (for example reassign `pipe.unet` and `pipe.text_encoder` to the adapted models, or load the saved weights into a fresh pipeline).
2. Confirm that the adapter weights changed during training (compare a LoRA matrix before and after) and that the training script optimises them.
3. Re-run `run_eval('lora')` and confirm that images differ from the baseline.
4. Raise the number of training words (relax the OCR check to 1 of 3 fonts, or check more candidate words).

## 7. How to run

1. Open in Google Colab, **Runtime → GPU (T4)**.
2. Run all cells in order. Settings (`N_TEST_WORDS`, `N_TRAIN_WORDS`, `GEN_STEPS`, `LORA_STEPS`, `LORA_RANK`, `TRAIN_TEXT_ENCODER`) are in the "Settings" cell.
3. Fonts are downloaded from the Google Fonts GitHub repository.

## 8. Outputs

Saved to `/content/glyph_hallucination/` and zipped as `glyph_hallucination_results.zip`:
- generated images for `baseline` and `lora`
- `test_words.csv`, `comparison.csv`
- plots generated by the notebook (`position_errors.png`, `side_by_side.png`)
- `arabic_glyph_lora.safetensors` (LoRA weights)

## 9. Next steps

1. Fix and re-run the LoRA evaluation, then report CER change with its paired bootstrap interval.
2. Longer phrases and sentences, and a second OCR engine to check that findings do not depend on one reader.
3. Newer text-to-image models and glyph-aware approaches for comparison.
4. Separate analysis of ligatures and diacritics.

## 10. Requirements

`kagglehub`, `diffusers`, `transformers`, `accelerate`, `peft`, `easyocr`, `arabic-reshaper`, `python-bidi`, PyTorch, pandas, matplotlib. The notebook installs them in its first cell.
