# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Ultra-low-latency fused PDF processing pipeline.

Key optimizations over the standard inprocess pipeline:
 1. Single process — all models loaded on the same GPU (no NIM/HTTP/gRPC)
 2. Parallel PDF rasterization — ThreadPool + pinned memory + async H→D
 3. Batched YOLOX — all pages in a single forward pass (not one-by-one)
 4. GPU-native region cropping — tensor slicing instead of PIL + base64
 5. Smart page routing — text-only pages skip OCR entirely
 6. Direct numpy handoff to OCR — bypass PNG/base64 encode-decode loop
 7. Optional torch.compile — JIT Triton kernels for model acceleration
 8. Pre-warmed CUDA context — eliminates cold-start penalty in timings
 9. No DataFrames between stages — raw dicts/tensors, build DF only at end
10. No Redis, no Ray, no queues — direct synchronous function calls
"""

from __future__ import annotations

import os
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Enable TF32 on Ampere+ for ~2× matmul throughput at negligible accuracy cost
torch.set_float32_matmul_precision("high")

try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

_STRUCTURED_LABELS = frozenset({"table", "chart", "infographic"})


@dataclass
class PageResult:
    """Per-page processing result."""

    page_number: int
    native_text: str
    is_scanned: bool
    orig_shape_hw: Tuple[int, int]
    num_detections: int = 0
    num_structured_regions: int = 0
    structured_texts: List[str] = field(default_factory=list)
    final_text: str = ""


@dataclass
class PDFResult:
    """Result from processing a single PDF."""

    source_path: str
    num_pages: int
    page_results: List[PageResult]
    all_texts: List[str]
    embeddings: Optional[torch.Tensor]
    timings: OrderedDict
    embedding_dim: int = 0


class FusedPDFPipeline:
    """
    Single-process, single-GPU fused pipeline.

    All models resident on one GPU.  Zero inter-process communication,
    zero serialization, zero base64 encoding between stages.  Tensors
    stay on GPU as long as possible.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        use_torch_compile: bool = False,
        render_threads: int = 8,
        embed_batch_size: int = 128,
        pe_batch_size: int = 32,
        score_threshold: float = 0.3,
    ):
        self.device = torch.device(device)
        self.render_threads = render_threads
        self.embed_batch_size = embed_batch_size
        self.pe_batch_size = pe_batch_size
        self.score_threshold = score_threshold
        self._use_compile = use_torch_compile
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def load_models(self) -> OrderedDict:
        """Load all models onto the target GPU.  Returns per-model load times."""
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
        from nemo_retriever.model import create_local_embedder

        self.embedder = create_local_embedder(device=str(self.device))
        times["embedder_s"] = time.perf_counter() - t0

        if self._use_compile:
            t0 = time.perf_counter()
            self._apply_torch_compile()
            times["torch_compile_s"] = time.perf_counter() - t0

        times["total_s"] = sum(times.values())
        self._models_loaded = True
        return times

    def _apply_torch_compile(self) -> None:
        # "default" mode: Triton JIT kernels without the expensive CUDA-graph
        # capture that "reduce-overhead" triggers (which adds 30-60 s warmup).
        compile_mode = "default"

        inner = getattr(self.pe_model, "_model", None)
        if inner is not None:
            backbone = getattr(inner, "model", inner)
            if isinstance(backbone, torch.nn.Module):
                try:
                    compiled = torch.compile(backbone, mode=compile_mode)
                    if hasattr(inner, "model"):
                        inner.model = compiled
                    else:
                        self.pe_model._model = compiled
                except Exception:
                    pass

        emb_model = getattr(self.embedder, "_model", None)
        if isinstance(emb_model, torch.nn.Module):
            try:
                self.embedder._model = torch.compile(emb_model, mode=compile_mode)
            except Exception:
                pass

    def warmup(self, runs: int = 2) -> float:
        """Pre-warm CUDA context and TRT/Triton caches.  Returns seconds."""
        if not self._models_loaded:
            raise RuntimeError("Call load_models() before warmup()")
        t0 = time.perf_counter()
        dummy = torch.randn(1, 3, 1024, 1024, device=self.device).clamp(0, 255)
        for _ in range(runs):
            with torch.inference_mode():
                try:
                    preds = self.pe_model.invoke(dummy, [(1024, 1024)])
                    self.pe_model.postprocess(preds)
                except Exception:
                    pass
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
        """Process one PDF end-to-end.  Returns embeddings + per-stage timings."""
        if not self._models_loaded:
            raise RuntimeError("Call load_models() first")

        timings: OrderedDict[str, float] = OrderedDict()
        src = os.path.abspath(pdf_path)

        # ── Stage 1: parallel page render + text extraction ──────────
        t0 = time.perf_counter()
        pages = self._render_pages(src)
        timings["1_render_ms"] = (time.perf_counter() - t0) * 1000

        n = len(pages)
        if n == 0:
            return PDFResult(src, 0, [], [], None, timings)

        # ── Stage 2: build GPU batch (pinned mem → async H→D) ────────
        t0 = time.perf_counter()
        batch, orig_shapes = self._to_gpu_batch(pages)
        torch.cuda.synchronize(self.device)
        timings["2_transfer_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 3: batched page-element detection ──────────────────
        t0 = time.perf_counter()
        boxes_l, labels_l, scores_l = self._detect_elements(batch, orig_shapes)
        torch.cuda.synchronize(self.device)
        timings["3_page_elements_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 4: classify pages + GPU crop structured regions ────
        t0 = time.perf_counter()
        page_results, crops = self._classify_and_crop(
            pages,
            batch,
            boxes_l,
            labels_l,
            scores_l,
        )
        timings["4_classify_crop_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 5: OCR structured regions (batched where possible) ─
        t0 = time.perf_counter()
        self._ocr_crops(page_results, crops)
        timings["5_ocr_ms"] = (time.perf_counter() - t0) * 1000

        # ── Stage 6: text assembly ───────────────────────────────────
        t0 = time.perf_counter()
        texts = self._assemble(page_results)
        timings["6_assemble_ms"] = (time.perf_counter() - t0) * 1000

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
            all_texts=texts,
            embeddings=emb,
            timings=timings,
            embedding_dim=int(emb.shape[1]) if emb is not None and emb.ndim == 2 else 0,
        )

    # ------------------------------------------------------------------
    # Stage 1 — PDF text extraction + rendering
    # ------------------------------------------------------------------

    def _render_pages(self, path: str) -> List[Dict[str, Any]]:
        if pdfium is None:
            raise ImportError("pypdfium2 is required")

        doc = pdfium.PdfDocument(path)
        n = len(doc)
        if n == 0:
            doc.close()
            return []

        # Phase A — sequentially extract text + split into independent
        # single-page byte buffers.  pdfium's C library is NOT thread-safe
        # for operations on the same document, so we must isolate pages
        # before handing them to worker threads.
        page_meta: List[Dict[str, Any]] = []
        for i in range(n):
            pg = doc[i]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                tp = pg.get_textpage()
                txt = tp.get_text_bounded()
                tp.close()

            # Save single-page PDF bytes so each thread gets its own doc
            single = pdfium.PdfDocument.new()
            single.import_pages(doc, pages=[i])
            buf = BytesIO()
            single.save(buf)
            single.close()
            pg.close()

            page_meta.append(
                {
                    "idx": i,
                    "text": txt,
                    "page_bytes": buf.getvalue(),
                }
            )
        doc.close()

        # Phase B — render each page in parallel.  Each worker opens its
        # own independent PdfDocument from the single-page bytes so there
        # is zero shared mutable state → no segfaults.
        def _render_one(meta: Dict[str, Any]) -> Dict[str, Any]:
            one_doc = pdfium.PdfDocument(meta["page_bytes"])
            pg = one_doc[0]
            pw, ph = float(pg.get_width()), float(pg.get_height())
            scale = min(1024.0 / max(pw, 1.0), 1024.0 / max(ph, 1.0))
            bmp = pg.render(scale=scale)
            pil = bmp.to_pil()
            arr = np.array(pil.convert("RGB"))
            bmp.close()
            pg.close()
            one_doc.close()
            return {
                "page_number": meta["idx"] + 1,
                "native_text": meta["text"],
                "is_scanned": len(meta["text"].strip()) < 10,
                "image_np": arr,
                "orig_hw": (int(arr.shape[0]), int(arr.shape[1])),
            }

        workers = max(1, min(self.render_threads, n))
        if workers > 1 and n > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_render_one, page_meta))
        else:
            results = [_render_one(m) for m in page_meta]

        return results

    # ------------------------------------------------------------------
    # Stage 2 — pinned-memory batch assembly + async transfer
    # ------------------------------------------------------------------

    def _to_gpu_batch(
        self,
        pages: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        orig_shapes: List[Tuple[int, int]] = []
        preprocessed: List[torch.Tensor] = []

        for p in pages:
            arr = p["image_np"]
            h, w = arr.shape[:2]
            orig_shapes.append((h, w))
            # HWC uint8 → CHW float32 (0-255 range)
            t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
            pp = self.pe_model.preprocess(t)  # resize_pad → [1, 3, 1024, 1024]
            preprocessed.append(pp)

        cpu_batch = torch.cat(preprocessed, dim=0)  # [N, 3, 1024, 1024]

        # Pinned memory → async H→D copy
        if torch.cuda.is_available():
            pinned = cpu_batch.pin_memory()
            gpu_batch = pinned.to(self.device, non_blocking=True)
        else:
            gpu_batch = cpu_batch.to(self.device)

        return gpu_batch, orig_shapes

    # ------------------------------------------------------------------
    # Stage 3 — batched YOLOX inference
    # ------------------------------------------------------------------

    def _detect_elements(
        self,
        batch: torch.Tensor,
        orig_shapes: List[Tuple[int, int]],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        all_boxes: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_scores: List[torch.Tensor] = []
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
                boxes = [boxes]
                labels = [labels]
                scores = [scores]
            all_boxes.extend(boxes)
            all_labels.extend(labels)
            all_scores.extend(scores)

        return all_boxes, all_labels, all_scores

    # ------------------------------------------------------------------
    # Stage 4 — classify pages + GPU-native region crop
    # ------------------------------------------------------------------

    def _classify_and_crop(
        self,
        pages: List[Dict[str, Any]],
        batch: torch.Tensor,
        boxes_list: List[torch.Tensor],
        labels_list: List[torch.Tensor],
        scores_list: List[torch.Tensor],
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

            # Page image on GPU: [3, H, W] (preprocessed 1024×1024)
            page_img = batch[i]
            _, h_img, w_img = page_img.shape

            n_det = int(boxes.shape[0]) if boxes.ndim >= 1 and boxes.numel() > 0 else 0
            pr.num_detections = n_det

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

                # Denormalize to pixel coords on the preprocessed image
                x1 = max(0, int(bx[0].item() * w_img))
                y1 = max(0, int(bx[1].item() * h_img))
                x2 = min(w_img, int(bx[2].item() * w_img))
                y2 = min(h_img, int(bx[3].item() * h_img))

                if x2 > x1 + 4 and y2 > y1 + 4:
                    # GPU tensor slice — zero-copy view, .contiguous() for OCR
                    crop = page_img[:, y1:y2, x1:x2].contiguous()
                    crops.append((i, crop, lbl))

            # Scanned pages with no structured content → OCR full page
            if p["is_scanned"] and pr.num_structured_regions == 0:
                crops.append((i, page_img.contiguous(), "full_page"))

            page_results.append(pr)

        return page_results, crops

    # ------------------------------------------------------------------
    # Stage 5 — OCR (direct numpy handoff, no PNG / base64 roundtrip)
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
                # GPU CHW float → CPU HWC uint8 numpy (skip PNG encode entirely)
                cpu_t = crop_gpu.cpu()
                if cpu_t.dtype.is_floating_point:
                    cpu_t = cpu_t.clamp(0, 255).to(torch.uint8)
                arr = cpu_t.permute(1, 2, 0).contiguous().numpy()  # HWC uint8

                result = self.ocr_model.invoke(arr, merge_level="paragraph")

                # The pipeline may return a list of dicts or a single object
                if isinstance(result, list):
                    text = " ".join(self.ocr_model._extract_text(item) for item in result).strip()
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
    # Stage 6 — text assembly
    # ------------------------------------------------------------------

    @staticmethod
    def _assemble(page_results: List[PageResult]) -> List[str]:
        texts: List[str] = []
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
        return texts

    # ------------------------------------------------------------------
    # Stage 7 — batched embedding
    # ------------------------------------------------------------------

    def _embed(self, texts: List[str]) -> Optional[torch.Tensor]:
        if not texts:
            return None
        prefixed = [f"passage: {t}" for t in texts]
        return self.embedder.embed(prefixed, batch_size=self.embed_batch_size)
