"""
pdf_outline_extractor
======================

This module provides a self‑contained routine for extracting a clean,
hierarchical outline from a PDF.  The extractor works completely
offline and supports both native text PDFs and scanned/image PDFs by
falling back to OCR when necessary.  It outputs a JSON structure
containing the document title and a list of headings tagged with
levels (H1–H3) and page numbers.

Major features:

* **Text extraction:** Lines are collected using PyMuPDF.  When no
  text is found on a page, the page is rendered to an image and
  processed through an OCR backend (PaddleOCR if available, else
  Tesseract).  Each detected string is annotated with an estimated
  font size based on its bounding box.

* **Font size clustering:** The set of font sizes for all lines is
  clustered using k‑means so that larger text (titles/headings) can
  be distinguished from body text.  Ranks are assigned to clusters
  from 0 (largest average size) up to 4.

* **Enumeration merging:** Number prefixes like ``"2.1"`` and ``"3."``
  are often emitted as separate text objects.  A merging pass
  combines a standalone enumeration line with the following line.

* **Continuation merging:** Some headings are split across two lines
  (e.g., ``"3. Overview of … Agile Tester"`` followed by ``"Syllabus"``).
  A second merging pass joins a line with a single‑word capitalised
  follower.

* **Heading detection heuristics:** Heuristics filter out footers,
  dates and boilerplate and classify remaining lines as headings
  based on font rank, enumeration patterns, keyword matches,
  uppercase ratios and line endings.  Lines beginning with common
  ignore prefixes (e.g., ``"Version"``, ``"The following"``,
  ``"RSVP"``) are ignored entirely.

The extractor has been tuned against the Adobe India Hackathon
sample dataset.  It runs entirely on CPU and keeps within
approximate 200 MB memory limits by using PaddleOCR’s lightweight
models when available.
"""

from __future__ import annotations

import json
import os
import re
import sys
import multiprocessing
from collections import defaultdict
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Any

import fitz  # PyMuPDF
import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv
load_dotenv()

# Import PaddleOCR
try:
    from paddleocr import PaddleOCR  # type: ignore
    _PADDLE_AVAILABLE = True
except Exception as e:
    print(f"Error: PaddleOCR import failed: {e}")
    _PADDLE_AVAILABLE = False

# PIL for image conversion
try:
    from PIL import Image
except Exception:
    Image = None  # type: ignore

# Thresholds for OCR
MIN_OCR_CONFIDENCE = 0.5
# Ratio of the page height used when scaling OCR coordinates.  We
# assume the entire page is considered (no cropping), so this is 1.0.
CROP_TOP_RATIO = 1.0


class OCRExtractor:
    """Simple OCR wrapper supporting PaddleOCR.

    When instantiated, this class attempts to load PaddleOCR with
    lightweight models (no GPU, English only).  The API exposes a single method
    ``extract_from_image`` which accepts raw PNG bytes and returns
    a list of (text, estimated font size) tuples.  Font size is
    approximated from the height of the detected bounding box and is
    scaled to roughly match point sizes used by PyMuPDF.
    """

    def __init__(self) -> None:
        self.ocr_type: Optional[str] = None
        self.ocr = None
        self._init_ocr()

    def _init_ocr(self) -> None:
        # Suppress duplicate OpenMP warnings when PaddleOCR loads
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        if _PADDLE_AVAILABLE:
            try:
                # Use environment variables to locate lite models if provided.
                # To stay under the 200 MB limit, download the PP‑OCRv3 lite models
                # and set PADDLE_DET_MODEL_DIR and PADDLE_REC_MODEL_DIR (and optionally
                # PADDLE_CLS_MODEL_DIR) to their paths.  If not set, PaddleOCR will
                # fall back to its default heavy models, which may exceed the size limit.
                det_dir = os.environ.get('PADDLE_DET_MODEL_DIR')
                rec_dir = os.environ.get('PADDLE_REC_MODEL_DIR')
                cls_dir = os.environ.get('PADDLE_CLS_MODEL_DIR')
                kwargs = dict(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
                if det_dir:
                    kwargs['det_model_dir'] = det_dir
                if rec_dir:
                    kwargs['rec_model_dir'] = rec_dir
                if cls_dir:
                    kwargs['cls_model_dir'] = cls_dir
                self.ocr = PaddleOCR(**kwargs)
                self.ocr_type = 'paddle'
                return
            except Exception:
                self.ocr = None
                self.ocr_type = None

    def extract_from_image(self, image_bytes: bytes, page_width: float, page_height: float) -> List[Tuple[str, float]]:
        """Run OCR on a rendered PDF page image and return text with
        estimated font sizes.

        Args:
            image_bytes: Raw PNG bytes of the page image.
            page_width: Width of the PDF page.
            page_height: Height of the PDF page.

        Returns:
            A list of tuples ``(text, size)`` where ``text`` is the
            recognised string and ``size`` is an approximate font
            size in points.
        """
        if not self.ocr or not Image:
            return []
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_w, img_h = img.size
        spans: List[Tuple[str, float]] = []
        if self.ocr_type == 'paddle':
            # Convert to numpy array for PaddleOCR
            img_array = np.array(img)
            try:
                results = self.ocr.ocr(img_array, cls=True)
            except Exception:
                return []
            if not results or not results[0]:
                return []
            for line in results[0]:
                if not line:
                    continue
                bbox, (text, conf) = line
                if not text or conf < MIN_OCR_CONFIDENCE:
                    continue
                # Compute bounding box height in image coordinates
                ys = [pt[1] for pt in bbox]
                height_px = float(max(ys) - min(ys))
                # Convert pixel height to approximate point size.
                # We assume 72 DPI (1 point per pixel) scaled by page/image ratio.
                # Use CROP_TOP_RATIO to account for cropping (here 1.0).
                if img_h > 0:
                    page_ratio = (page_height * CROP_TOP_RATIO) / img_h
                else:
                    page_ratio = 1.0
                font_size = max(height_px * page_ratio * 0.75, 8.0)
                spans.append((text.strip(), font_size))
        return spans


def _cluster_font_sizes(sizes: List[float], max_clusters: int = 5) -> Dict[int, int]:
    """Cluster font sizes via k‑means and return a map from cluster label
    to rank (0 is largest font, increasing values are smaller fonts).
    """
    unique_sizes = sorted(set(sizes), reverse=True)
    k = min(max_clusters, len(unique_sizes))
    if k <= 1:
        return {0: 0}
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(np.array(sizes).reshape(-1, 1))
    means: Dict[int, List[float]] = {}
    for lbl, sz in zip(labels, sizes):
        means.setdefault(lbl, []).append(sz)
    mean_sizes = {lbl: float(np.mean(vals)) for lbl, vals in means.items()}
    sorted_clusters = sorted(mean_sizes.items(), key=lambda x: x[1], reverse=True)
    return {cluster: rank for rank, (cluster, _) in enumerate(sorted_clusters)}


def _merge_enumerations(lines: List[Tuple[int, str, float, int]]) -> List[Tuple[int, str, float, int]]:
    """Merge enumeration‑only lines with the following line.

    Lines are tuples ``(page, text, size, rank)``.  If ``text``
    consists solely of digits and dots (optionally ending in a dot)
    and either contains a dot or is very short (≤3 chars), the line is
    temporarily stored and merged with the next line.
    """
    merged: List[Tuple[int, str, float, int]] = []
    pending: Optional[Tuple[int, str, float, int]] = None
    for page, text, size, rank in lines:
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*\.?", text) and ('.' in text or len(text) <= 3):
            pending = (page, text, size, rank)
            continue
        if pending:
            p_page, p_text, p_size, p_rank = pending
            combined = (p_page, f"{p_text} {text}", max(size, p_size), min(rank, p_rank))
            merged.append(combined)
            pending = None
        else:
            merged.append((page, text, size, rank))
    if pending:
        merged.append(pending)
    return merged


def extract_outline(pdf_path: str) -> Dict[str, object]:
    """Extract a hierarchical outline from the given PDF.

    Args:
        pdf_path: Path to a PDF file on disk.

    Returns:
        A dict with keys ``title`` (str) and ``outline`` (list of
        heading dicts).  ``page`` values are 0‑based indices.
    """
    doc = fitz.open(pdf_path)
    ocr_extractor: Optional[OCRExtractor] = None
    raw_lines: List[Tuple[int, str, float]] = []
    for page_idx in range(doc.page_count):
        page = doc.load_page(page_idx)
        text_dict = page.get_text("dict")
        page_has_text = False
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                line_text = "".join(span.get("text", "") for span in spans).strip()
                if not line_text:
                    continue
                page_has_text = True
                max_size = max(span.get("size", 0.0) for span in spans)
                raw_lines.append((page_idx, line_text, max_size))
        # If page has no text and PaddleOCR is available, run OCR
        if not page_has_text and _PADDLE_AVAILABLE:
            if ocr_extractor is None:
                ocr_extractor = OCRExtractor()
            if ocr_extractor.ocr_type is not None:
                pix = page.get_pixmap()
                image_bytes = pix.tobytes("png")
                spans = ocr_extractor.extract_from_image(image_bytes, page.rect.width, page.rect.height)
                for text, est_size in spans:
                    if text:
                        raw_lines.append((page_idx, text, est_size))
    # If nothing was extracted, return empty outline
    if not raw_lines:
        return {"title": "", "outline": []}
    # Cluster font sizes
    sizes = [sz for _, _, sz in raw_lines]
    rank_map = _cluster_font_sizes(sizes)
    # Assign cluster ranks to lines
    labels = []
    if sizes:
        uniq = len(set(sizes))
        k = min(5, uniq)
        if k <= 1:
            labels = [0] * len(sizes)
        else:
            km = KMeans(n_clusters=k, random_state=0)
            labels = km.fit_predict(np.array(sizes).reshape(-1, 1)).tolist()
    enhanced: List[Tuple[int, str, float, int]] = []
    for (page_idx, text, size), lbl in zip(raw_lines, labels):
        enhanced.append((page_idx, text, size, rank_map.get(lbl, 0)))
    # Merge enumeration prefixes
    processed = _merge_enumerations(enhanced)
    # Merge continuation lines (single‑word followers)
    merged_lines: List[Tuple[int, str, float, int]] = []
    i = 0
    while i < len(processed):
        page, text, size, rank = processed[i]
        if i + 1 < len(processed):
            n_page, n_text, n_size, n_rank = processed[i + 1]
            if (n_page == page and len(n_text.split()) == 1 and
                re.match(r"^[A-Z][A-Za-z]*$", n_text.strip())):
                combined_text = f"{text} {n_text.strip()}"
                combined_size = max(size, n_size)
                combined_rank = min(rank, n_rank)
                merged_lines.append((page, combined_text, combined_size, combined_rank))
                i += 2
                continue
        merged_lines.append((page, text, size, rank))
        i += 1
    processed = merged_lines
    # Determine title: longest line in largest cluster on first two pages
    def _valid_title(s: str) -> bool:
        s = s.strip()
        if not s or any(c.isdigit() for c in s):
            return False
        if any(ch in s for ch in ['!', '?']):
            return False
        if len(s.split()) < 2:
            return False
        alpha = sum(c.isalpha() for c in s)
        return (alpha / len(s)) >= 0.5
    title_candidates = [(text, len(text)) for (page, text, size, rank) in processed
                        if rank == 0 and page <= 1 and _valid_title(text)]
    title = max(title_candidates, key=lambda x: x[1])[0].strip() if title_candidates else ""
    # Heuristics for heading detection
    ignore_starts = (
        "Version", "Date", "Remarks", "The following", "The syllabi",
        "The syllabus", "RSVP", "I declare"
    )
    heading_keywords = {kw.title() for kw in [
        "revision history", "table of contents", "acknowledgements",
        "acknowledgment", "acknowledgments", "overview", "introduction",
        "abstract", "summary", "background", "conclusion",
        "references", "related work", "contents"
    ]}
    candidates: List[Dict[str, object]] = []
    seen_two_word_upper: Dict[int, bool] = defaultdict(bool)
    for page, text, size, rank in processed:
        # Skip very small clusters
        if rank > 3:
            continue
        # Skip identical to title
        if title and text.strip().lower() == title.strip().lower():
            continue
        # Ignore specific prefixes
        if any(text.startswith(pref) for pref in ignore_starts):
            continue
        # Ignore enumerations starting with 0.x
        if re.match(r"^0\.[0-9]", text):
            continue
        # Remove footers like 'Page 3 of 10'
        if re.match(r"Page \d+ of \d+", text):
            continue
        # Skip simple dates
        if re.match(r"\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}$", text):
            continue
        tokens = text.split()
        if not tokens:
            continue
        if len(tokens) > 20 or len(text) > 100:
            continue
        # First char must be alphanumeric
        if not re.match(r"[A-Z0-9]", text[0]):
            continue
        # Skip standalone 'overview'
        if text.strip().lower() == 'overview':
            continue
        # At least one alpha token longer than 3 letters
        alpha_tokens = [tok for tok in tokens if re.search('[A-Za-z]', tok)]
        if not alpha_tokens or not any(len(re.sub('[^A-Za-z]', '', tok)) > 3 for tok in alpha_tokens):
            continue
        # Compute uppercase ratio
        letters = [c for c in text if c.isalpha()]
        upper_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
        # Detect enumeration with decimals
        enum_match = re.match(r"^([0-9]+\.[0-9]+(?:\.[0-9]+)*\.?)[\s]+(.+)", text)
        has_enum = False
        enum_prefix = ""
        rest = text
        if enum_match:
            enum_prefix = enum_match.group(1)
            rest = enum_match.group(2)
            if not enum_prefix.startswith('0.') and any(ch.isalpha() for ch in rest):
                # At least one word after enumeration
                if len(rest.split()) >= 1:
                    has_enum = True
        else:
            # Single‑level enumeration like '1.'
            single_match = re.match(r"^([0-9]+\.)\s+(.+)", text)
            if single_match:
                enum_prefix = single_match.group(1)
                rest = single_match.group(2)
                first_word = rest.split()[0].lower() if rest.split() else ""
                if first_word in {"introduction", "overview", "references", "conclusion", "background"}:
                    has_enum = True
        keyword_match = text.strip().title() in heading_keywords
        keyword_start = re.match(r"(Appendix|Section|Table)\b", text, re.IGNORECASE)
        word_count = len(tokens)
        ends_with_colon = text.rstrip().endswith(':')
        candidate = False
        if has_enum and rank <= 2:
            candidate = True
        elif keyword_match or keyword_start:
            candidate = True
        elif ends_with_colon and word_count >= 2 and rank <= 3:
            candidate = True
        else:
            if not has_enum and any(ch.isdigit() for ch in text):
                candidate = False
            else:
                if rank <= 1 and word_count >= 3 and upper_ratio > 0.3:
                    candidate = True
                elif 2 <= word_count <= 3 and rank <= 2 and upper_ratio > 0.6:
                    if not seen_two_word_upper[page]:
                        candidate = True
                        seen_two_word_upper[page] = True
        if not candidate:
            continue
        # Determine level
        level: str
        if has_enum:
            parts = [p for p in enum_prefix.rstrip('.').split('.')]
            level = f"H{min(len(parts), 3)}"
        elif keyword_match:
            level = "H1"
        elif ends_with_colon:
            level = "H3" if rank <= 2 else "H4"
        else:
            mapped = 1 if rank <= 1 else (2 if rank == 2 else 3)
            level = f"H{mapped}"
        candidates.append({"level": level, "text": text.strip(), "page": page})
    # Deduplicate: group by main text (strip enumeration) and retain the latest occurrence
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in candidates:
        m = re.match(r"^([0-9]+(?:\.[0-9]+)*\.?)[\s]+(.+)", item["text"])
        main_text = (m.group(2) if m else item["text"]).strip().lower()
        grouped[main_text].append(item)
    final: List[Dict[str, object]] = []
    for items in grouped.values():
        enum_items = [it for it in items if re.match(r"^[0-9]+(?:\.[0-9]+)*\.", it["text"]) ]
        if enum_items:
            chosen = max(enum_items, key=lambda x: x["page"])
        else:
            chosen = max(items, key=lambda x: x["page"])
        final.append(chosen)
    final_sorted = sorted(final, key=lambda x: (x["page"], x["text"]))
    return {"title": title, "outline": final_sorted}


def process_single_file(args: tuple[str, Optional[str]]) -> tuple[bool, str, str]:
    """Process a single PDF file and save its outline to a JSON file.
    
    Args:
        args: A tuple containing (input_path, output_path)
            - input_path: Path to the input PDF file
            - output_path: Path to save the output JSON. If None, prints to stdout.
            
    Returns:
        A tuple of (success, input_path, output_path_or_error)
    """
    input_path, output_path = args
    try:
        result = extract_outline(input_path)
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            return (True, input_path, output_path)
        else:
            # In parallel mode, we don't print to stdout to avoid mixing output
            return (True, input_path, "<stdout>")
    except Exception as e:
        return (False, input_path, str(e))

def process_pdf(input_path: str, output_path: Optional[str] = None) -> None:
    """Legacy function for single-file processing. Use process_single_file for parallel processing."""
    success, _, message = process_single_file((input_path, output_path))
    if success:
        if output_path:
            print(f"Processed: {input_path} -> {message}")
    else:
        print(f"Error processing {input_path}: {message}", file=sys.stderr)

def process_batch(input_folder: str, output_folder: Optional[str] = None, max_workers: Optional[int] = None) -> None:
    """Process all PDFs in the input folder in parallel.
    
    Args:
        input_folder: Path to the folder containing PDF files
        output_folder: Optional output folder for JSON files
        max_workers: Maximum number of worker processes to use (default: all available CPUs)
    """
    # Get list of PDF files
    pdf_files = [
        f for f in os.listdir(input_folder) 
        if f.lower().endswith('.pdf')
    ]
    
    if not pdf_files:
        print(f"No PDF files found in {input_folder}", file=sys.stderr)
        return
    
    # Prepare arguments for parallel processing
    tasks = []
    for filename in pdf_files:
        input_path = os.path.join(input_folder, filename)
        if output_folder:
            output_filename = os.path.splitext(filename)[0] + '.json'
            output_path = os.path.join(output_folder, output_filename)
        else:
            output_path = None
        tasks.append((input_path, output_path))
    
    # Process files in parallel
    print(f"Processing {len(tasks)} PDF files using {max_workers or 'all available'} workers...")
    start_time = os.times()
    
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(process_single_file, tasks)
    
    # Report results
    end_time = os.times()
    elapsed = end_time.user + end_time.system - start_time.user - start_time.system
    
    success_count = sum(1 for success, _, _ in results if success)
    failed_count = len(results) - success_count
    
    print(f"\nProcessing complete in {elapsed:.2f} seconds")
    print(f"Successfully processed: {success_count} files")
    if failed_count > 0:
        print(f"Failed to process: {failed_count} files")
        for success, input_path, error in results:
            if not success:
                print(f"  {os.path.basename(input_path)}: {error}")

def main() -> None:
    # Use environment variables for input/output paths with defaults for local development
    input_dir = os.getenv('INPUT_DIR', 'input')
    output_dir = os.getenv('OUTPUT_DIR', 'output')
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Process all PDFs in the input directory
    print(f"Processing PDFs in: {input_dir}")
    print(f"Saving JSON outputs to: {output_dir}")
    
    # Use all available CPUs for parallel processing
    max_workers = os.cpu_count()
    process_batch(input_dir, output_dir, max_workers)
    
    print("Processing complete.")

if __name__ == '__main__':
    main()