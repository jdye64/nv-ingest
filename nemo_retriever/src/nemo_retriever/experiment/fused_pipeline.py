# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ultra-low-latency fused PDF processing pipeline.

Key optimizations over the standard inprocess pipeline:
 1. Single process — all models loaded on the same GPU (no NIM/HTTP/gRPC)
 2. Batched YOLOX — all pages in a single forward pass (not one-by-one)
 3. GPU-native region cropping — tensor slicing instead of PIL + base64
 4. Smart OCR routing — aggressive filtering to minimize OCR invocations
 5. Direct numpy handoff to OCR — bypass PNG/base64 encode-decode loop
 6. Coarse chunking — merge page texts to cut embedding count by 5-10x
 7. Fast embedder option — sentence-transformers ~33-110M instead of 1B
 8. TF32 + cuDNN benchmark — free speedups on Ampere+
 9. Pinned memory + async H→D transfer
10. No DataFrames, no Redis, no Ray, no queues
"""

from __future__ import annotations

import os
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_float32_matmul_precision("high")

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

_STRUCTURED_LABELS = frozenset({"table", "chart", "infographic"})


# ======================================================================
# Data classes
# ======================================================================


@dataclass
class PageResult:
    page_number: int
    native_text: str
    is_scanned: bool
    orig_shape_hw: Tuple[int, int]
    num_detections: int = 0
    num_structured_regions: int = 0
    ocr_skipped: bool = False
    structured_texts: List[str] = field(default_factory=list)
    final_text: str = ""


@dataclass
class PDFResult:
    source_path: str
    num_pages: int
    page_results: List[PageResult]
    chunks: List[Any]  # List[ChunkInfo] — forward ref
    all_texts: List[str]
    embeddings: Optional[torch.Tensor]
    timings: OrderedDict
    embedding_dim: int = 0
    ocr_stats: Dict[str, int] = field(default_factory=dict)


# ======================================================================
# Lightweight embedder — sentence-transformers via HuggingFace
# ======================================================================


@dataclass
class FastEmbedder:
    """Minimal wrapper around a small HF sentence-transformer.

    Loads any model that follows the standard AutoModel + mean-pooling
    pattern (e5-small, snowflake-arctic-embed-m, etc.).
    """

    model_id: str = "intfloat/e5-small-v2"
    device: str = "cuda:0"
    max_length: int = 512
    normalize: bool = True

    def __post_init__(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        dev = torch.device(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModel.from_pretrained(self.model_id)
        self._model = self._model.to(dev).eval()
        self._device = dev

    @property
    def is_remote(self) -> bool:
        return False

    def embed(self, texts: list[str], *, batch_size: int = 256) -> torch.Tensor:
        texts_list = [str(t) for t in texts if str(t).strip()]
        if not texts_list:
            return torch.empty((0, 0), dtype=torch.float32)

        outs: list[torch.Tensor] = []
        with torch.inference_mode(), torch.autocast(device_type="cuda"):
            for i in range(0, len(texts_list), batch_size):
                chunk = texts_list[i : i + batch_size]
                batch = self._tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self._device)
                out = self._model(**batch)
                lhs = out.last_hidden_state  # [B, S, D]
                mask = batch["attention_mask"].unsqueeze(-1)
                vec = (lhs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
                vec = vec.detach().cpu().float()
                if self.normalize:
                    vec = vec / vec.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                outs.append(vec)

        return torch.cat(outs, dim=0) if outs else torch.empty((0, 0), dtype=torch.float32)


# ======================================================================
# Coarse chunker — merge per-page texts into larger windows
# ======================================================================


@dataclass
class ChunkInfo:
    """A text chunk with metadata about which pages it spans."""
    text: str
    page_numbers: List[int]


def coarse_chunk(
    page_texts: List[str],
    page_numbers: List[int],
    target_chars: int = 2000,
    overlap_chars: int = 200,
) -> List[ChunkInfo]:
    """Merge per-page texts into larger chunks to reduce embedding count.

    Returns ``ChunkInfo`` objects that track which page numbers each
    chunk covers — needed for recall evaluation against page-level GT.
    """
    if not page_texts:
        return []

    chunks: List[ChunkInfo] = []
    buf_texts: List[str] = []
    buf_pages: List[int] = []
    buf_len = 0

    for txt, pn in zip(page_texts, page_numbers):
        t = txt.strip()
        if not t:
            continue
        buf_texts.append(t)
        buf_pages.append(pn)
        buf_len += len(t)

        if buf_len >= target_chars:
            merged = "\n\n".join(buf_texts)
            chunks.append(ChunkInfo(text=merged, page_numbers=list(buf_pages)))
            tail = merged[-overlap_chars:] if overlap_chars and len(merged) > overlap_chars else ""
            # Overlap carries last page's number for matching
            buf_texts = [tail] if tail else []
            buf_pages = [buf_pages[-1]] if tail else []
            buf_len = len(tail)

    if buf_texts:
        merged = "\n\n".join(buf_texts)
        if merged.strip():
            chunks.append(ChunkInfo(text=merged, page_numbers=list(buf_pages)))

    return chunks


# ======================================================================
# Main pipeline
# ======================================================================


class FusedPDFPipeline:
    """Single-process, single-GPU fused pipeline with aggressive OCR
    filtering, coarse chunking, and optional fast embedder."""

    def __init__(
        self,
        device: str = "cuda:0",
        embed_batch_size: int = 128,
        pe_batch_size: int = 32,
        # --- OCR filtering knobs ---
        score_threshold: float = 0.3,
        min_crop_px: int = 32,
        max_crops_per_page: int = 5,
        skip_ocr_if_text: bool = True,
        # --- Chunking knobs ---
        chunk_target_chars: int = 2000,
        chunk_overlap_chars: int = 200,
        # --- Embedder choice ---
        use_fast_embedder: bool = False,
        fast_embedder_model: str = "intfloat/e5-small-v2",
    ):
        self.device = torch.device(device)
        self.embed_batch_size = embed_batch_size
        self.pe_batch_size = pe_batch_size
        self.score_threshold = score_threshold
        self.min_crop_px = min_crop_px
        self.max_crops_per_page = max_crops_per_page
        self.skip_ocr_if_text = skip_ocr_if_text
        self.chunk_target_chars = chunk_target_chars
        self.chunk_overlap_chars = chunk_overlap_chars
        self._use_fast_embedder = use_fast_embedder
        self._fast_embedder_model = fast_embedder_model
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> OrderedDict:
        torch.backends.cudnn.benchmark = True
        times: OrderedDict[str, float] = OrderedDict()

        t0 = time.perf_counter()
        from nemo_retriever.model.local import NemotronPageElementsV3
        self.pe_model = NemotronPageElementsV3()
        times["page_elements_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        from nemo_retriever.model.local import NemotronOCRV1
        self.ocr_model = NemotronOCRV1()
        times["ocr_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        if self._use_fast_embedder:
            self.embedder = FastEmbedder(
                model_id=self._fast_embedder_model,
                device=str(self.device),
            )
            times["embedder_s (fast)"] = time.perf_counter() - t0
        else:
            from nemo_retriever.model import create_local_embedder
            self.embedder = create_local_embedder(device=str(self.device))
            times["embedder_s (1B)"] = time.perf_counter() - t0

        times["total_s"] = sum(times.values())
        self._models_loaded = True
        return times

    def warmup(self, runs: int = 2) -> float:
        if not self._models_loaded:
            raise RuntimeError("Call load_models() before warmup()")
        t0 = time.perf_counter()

        dummy = torch.randn(1, 3, 1024, 1024, device=self.device).clamp(0, 255)
        for _ in range(runs):
            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                try:
                    preds = self.pe_model.invoke(dummy, [(1024, 1024)])
                    self.pe_model.postprocess(preds)
                except Exception:
                    pass

        for _ in range(runs):
            try:
                self.embedder.embed(["warmup text passage"], batch_size=1)
            except Exception:
                pass

        torch.cuda.synchronize(self.device)
        return time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str) -> PDFResult:
        if not self._models_loaded:
            raise RuntimeError("Call load_models() first")

        timings: OrderedDict[str, float] = OrderedDict()
        src = os.path.abspath(pdf_path)
        ocr_stats = {"crops_sent": 0, "crops_skipped": 0, "pages_skipped_ocr": 0}

        # ── Stage 1: render + text extraction ────────────────────────
        t0 = time.perf_counter()
        pages = self._render_pages(src)
        timings["1_render_ms"] = (time.perf_counter() - t0) * 1000

        n = len(pages)
        if n == 0:
            return PDFResult(src, 0, [], [], [], None, timings)

        # ── Stage 2: GPU batch ───────────────────────────────────────
        t0 = time.perf_counter()
        batch, orig_shapes = self._to_gpu_batch(pages)
        torch.cuda.synchronize(self.device)
        timings["2_transfer_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 3: batched page-element detection ──────────────────
        t0 = time.perf_counter()
        boxes_l, labels_l, scores_l = self._detect_elements(batch, orig_shapes)
        torch.cuda.synchronize(self.device)
        timings["3_page_elements_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: smart classify + crop (with aggressive filtering)
        t0 = time.perf_counter()
        page_results, crops = self._classify_and_crop(
            pages, batch, boxes_l, labels_l, scores_l, ocr_stats,
        )
        timings["4_classify_crop_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 5: OCR (only the crops that survived filtering) ────
        t0 = time.perf_counter()
        self._ocr_crops(page_results, crops)
        ocr_stats["crops_sent"] = len(crops)
        timings["5_ocr_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 6: text assembly ───────────────────────────────────
        t0 = time.perf_counter()
        per_page_texts, per_page_numbers = self._assemble(page_results)
        timings["6_assemble_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 6b: coarse chunking ────────────────────────────────
        t0 = time.perf_counter()
        chunk_infos = coarse_chunk(
            per_page_texts,
            per_page_numbers,
            target_chars=self.chunk_target_chars,
            overlap_chars=self.chunk_overlap_chars,
        )
        texts = [c.text for c in chunk_infos]
        timings["6b_chunk_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 7: batched embedding ───────────────────────────────
        t0 = time.perf_counter()
        emb = self._embed(texts)
        torch.cuda.synchronize(self.device)
        timings["7_embed_ms"] = (time.perf_counter() - t0) * 1000

        timings["total_ms"] = sum(timings.values())

        return PDFResult(
            source_path=src,
            num_pages=n,
            page_results=page_results,
            chunks=chunk_infos,
            all_texts=texts,
            embeddings=emb,
            timings=timings,
            embedding_dim=int(emb.shape[1]) if emb is not None and emb.ndim == 2 else 0,
            ocr_stats=ocr_stats,
        )

    # ------------------------------------------------------------------
    # Stage 1 — sequential PDF rendering
    # ------------------------------------------------------------------

    def _render_pages(self, path: str) -> List[Dict[str, Any]]:
        if pdfium is None:
            raise ImportError("pypdfium2 is required")

        doc = pdfium.PdfDocument(path)
        n = len(doc)
        if n == 0:
            doc.close()
            return []

        results: List[Dict[str, Any]] = []
        for i in range(n):
            pg = doc[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tp = pg.get_textpage()
                try:
                    txt = tp.get_text_bounded()
                except (TypeError, AttributeError):
                    txt = tp.get_text_range()
                tp.close()

            pw, ph = float(pg.get_width()), float(pg.get_height())
            scale = min(1024.0 / max(pw, 1.0), 1024.0 / max(ph, 1.0))
            bmp = pg.render(scale=scale)
            pil = bmp.to_pil()
            arr = np.array(pil.convert("RGB"))
            bmp.close()
            pg.close()

            results.append({
                "page_number": i + 1,
                "native_text": txt,
                "is_scanned": len(txt.strip()) < 10,
                "image_np": arr,
                "orig_hw": (int(arr.shape[0]), int(arr.shape[1])),
            })

        doc.close()
        return results

    # ------------------------------------------------------------------
    # Stage 2 — batch assembly
    # ------------------------------------------------------------------

    def _to_gpu_batch(
        self, pages: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        orig_shapes: List[Tuple[int, int]] = []
        preprocessed: List[torch.Tensor] = []

        for p in pages:
            arr = p["image_np"]
            h, w = arr.shape[:2]
            orig_shapes.append((h, w))
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
            pp = self.pe_model.preprocess(t)
            preprocessed.append(pp)

        cpu_batch = torch.cat(preprocessed, dim=0)

        if torch.cuda.is_available():
            pinned = cpu_batch.pin_memory()
            gpu_batch = pinned.to(self.device, non_blocking=True)
        else:
            gpu_batch = cpu_batch.to(self.device)

        return gpu_batch, orig_shapes

    # ------------------------------------------------------------------
    # Stage 3 — batched YOLOX
    # ------------------------------------------------------------------

    def _detect_elements(
        self, batch: torch.Tensor, orig_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        all_boxes, all_labels, all_scores = [], [], []
        n = batch.shape[0]
        bs = self.pe_batch_size

        for start in range(0, n, bs):
            end = min(start + bs, n)
            chunk = batch[start:end]
            chunk_shapes = orig_shapes[start:end]

            with torch.inference_mode(), torch.autocast(device_type="cuda"):
                preds = self.pe_model.invoke(chunk, chunk_shapes)

            boxes, labels, scores = self.pe_model.postprocess(preds)
            if not isinstance(boxes, list):
                boxes, labels, scores = [boxes], [labels], [scores]
            all_boxes.extend(boxes)
            all_labels.extend(labels)
            all_scores.extend(scores)

        return all_boxes, all_labels, all_scores

    # ------------------------------------------------------------------
    # Stage 4 — smart classify + crop with aggressive filtering
    #
    # Three OCR-reduction strategies:
    #   (a) skip_ocr_if_text: if a page already has rich native text
    #       (> 50 chars), skip OCR on its structured regions entirely.
    #   (b) max_crops_per_page: cap the number of OCR calls per page.
    #       Keep only the highest-confidence detections.
    #   (c) min_crop_px: ignore tiny crops that are likely noise.
    # ------------------------------------------------------------------

    def _classify_and_crop(
        self,
        pages: List[Dict[str, Any]],
        batch: torch.Tensor,
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        scores_list: List[torch.Tensor],
        ocr_stats: Dict[str, int],
    ) -> Tuple[List[PageResult], List[Tuple[int, torch.Tensor, str]]]:
        model_labels = self.pe_model.model.labels
        page_results: List[PageResult] = []
        crops: List[Tuple[int, torch.Tensor, str]] = []

        for i, p in enumerate(pages):
            boxes = boxes_list[i] if i < len(boxes_list) else torch.empty(0, 4)
            labels = labels_list[i] if i < len(labels_list) else torch.empty(0, dtype=torch.int64)
            scores = scores_list[i] if i < len(scores_list) else torch.empty(0)

            pr = PageResult(
                page_number=p["page_number"],
                native_text=p["native_text"],
                is_scanned=p["is_scanned"],
                orig_shape_hw=p["orig_hw"],
            )

            page_img = batch[i]
            _, h_img, w_img = page_img.shape

            n_det = int(boxes.shape[0]) if boxes.ndim >= 1 and boxes.numel() > 0 else 0
            pr.num_detections = n_det

            has_rich_text = len(p["native_text"].strip()) > 50

            # (a) If the page has rich native text, skip OCR entirely
            if self.skip_ocr_if_text and has_rich_text and not p["is_scanned"]:
                pr.ocr_skipped = True
                ocr_stats["pages_skipped_ocr"] = ocr_stats.get("pages_skipped_ocr", 0) + 1
                page_results.append(pr)
                continue

            # Collect candidate crops with their scores for ranking
            candidates: List[Tuple[float, torch.Tensor, str]] = []

            for j in range(n_det):
                label_idx = int(labels[j])
                if label_idx < 0 or label_idx >= len(model_labels):
                    continue
                lbl = model_labels[label_idx].lower()
                score = float(scores[j])

                if lbl not in _STRUCTURED_LABELS or score < self.score_threshold:
                    continue

                pr.num_structured_regions += 1
                box = boxes[j]
                bx = box.cpu().float() if isinstance(box, torch.Tensor) else torch.tensor(box)

                x1 = max(0, int(bx[0].item() * w_img))
                y1 = max(0, int(bx[1].item() * h_img))
                x2 = min(w_img, int(bx[2].item() * w_img))
                y2 = min(h_img, int(bx[3].item() * h_img))

                crop_w, crop_h = x2 - x1, y2 - y1

                # (c) Skip tiny crops
                if crop_w < self.min_crop_px or crop_h < self.min_crop_px:
                    ocr_stats["crops_skipped"] = ocr_stats.get("crops_skipped", 0) + 1
                    continue

                crop = page_img[:, y1:y2, x1:x2].contiguous()
                candidates.append((score, crop, lbl))

            # (b) Keep only top-K crops by confidence
            candidates.sort(key=lambda x: x[0], reverse=True)
            kept = candidates[: self.max_crops_per_page]
            ocr_stats["crops_skipped"] = ocr_stats.get("crops_skipped", 0) + (len(candidates) - len(kept))

            for score, crop, lbl in kept:
                crops.append((i, crop, lbl))

            # Scanned pages with no structured content → OCR full page
            if p["is_scanned"] and len(kept) == 0:
                crops.append((i, page_img.contiguous(), "full_page"))

            page_results.append(pr)

        return page_results, crops

    # ------------------------------------------------------------------
    # Stage 5 — OCR
    # ------------------------------------------------------------------

    def _ocr_crops(
        self,
        page_results: List[PageResult],
        crops: List[Tuple[int, torch.Tensor, str]],
    ) -> None:
        if not crops:
            return

        for page_idx, crop_gpu, label in crops:
            try:
                cpu_t = crop_gpu.cpu()
                if cpu_t.dtype.is_floating_point:
                    cpu_t = cpu_t.clamp(0, 255).to(torch.uint8)
                arr = cpu_t.permute(1, 2, 0).contiguous().numpy()

                result = self.ocr_model.invoke(arr, merge_level="paragraph")

                if isinstance(result, list):
                    text = " ".join(
                        self.ocr_model._extract_text(item) for item in result
                    ).strip()
                else:
                    text = self.ocr_model._extract_text(result)

                if not text:
                    continue

                if label == "full_page":
                    page_results[page_idx].native_text = text
                else:
                    page_results[page_idx].structured_texts.append(text)
            except Exception as e:
                print(f"  OCR warning (page {page_idx}, {label}): {e}")

    # ------------------------------------------------------------------
    # Stage 6 — text assembly (per-page texts)
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble(page_results: List[PageResult]) -> Tuple[List[str], List[int]]:
        texts: List[str] = []
        page_nums: List[int] = []
        for pr in page_results:
            parts: List[str] = []
            nt = pr.native_text.strip()
            if nt:
                parts.append(nt)
            for st in pr.structured_texts:
                s = st.strip()
                if s:
                    parts.append(s)
            final = "\n\n".join(parts)
            pr.final_text = final
            if final:
                texts.append(final)
                page_nums.append(pr.page_number)
        return texts, page_nums

    # ------------------------------------------------------------------
    # Stage 7 — batched embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> Optional[torch.Tensor]:
        if not texts:
            return None
        prefixed = [f"passage: {t}" for t in texts]
        return self.embedder.embed(prefixed, batch_size=self.embed_batch_size)
