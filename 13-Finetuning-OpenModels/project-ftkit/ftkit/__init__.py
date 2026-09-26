"""ftkit — paket stdlib murni untuk project Bab 13 (Fine-tuning & Open-Weight Models).

Modul:
    data.py      : dataset sentimen, format SFT, tarif, varian kuantisasi (JANGAN DIUBAH)
    tokenizer.py : tokenizer mini deterministik (vocab + OOV + negasi)      — GIVEN
    model.py     : klasifier tiny + loss BCE + gradien analitik             — GIVEN
    sft.py       : TODO: dataset JSONL ala Alpaca + oversampling
    lora.py      : TODO: matematika LoRA (A@B) + gradient check
    quantize.py  : TODO: simulasi kuantisasi & kompresi
    trainer.py   : TODO: loop SFT + early stopping + riwayat
    api.py       : TODO: biaya API vs lokal + perbandingan varian
    evaluasi.py  : TODO: eval sebelum/sesudah + kebijakan OOD

Prasyarat: stdlib saja (math, json). Semua kontrak ada di tests/.
Kunci angka: SEED 13 — loss, akurasi, epoch, rasio param semuanya deterministik.
"""
__all__ = ["data", "tokenizer", "model", "sft", "lora",
           "quantize", "trainer", "api", "evaluasi"]
