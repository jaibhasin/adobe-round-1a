# PDF Outline Extractor

A Docker-based solution for extracting hierarchical outlines from PDF documents using PaddleOCR and PyMuPDF. This tool is designed to work in offline environments with CPU-only support.

## Features

- Extracts hierarchical document structure from PDFs
- Supports both text-based and scanned PDFs using OCR
- Runs in a containerized environment
- No internet connection required during execution
- Optimized for AMD64 architecture
- Optimized for CPU usage with small memory footprint
- Processes multiple PDFs in parallel for better performance
- Outputs structured JSON with heading levels and page numbers

## Technical Approach

1. **Text Extraction**:
   - Uses PyMuPDF for native text extraction from PDFs
   - Falls back to OCR (PaddleOCR or Tesseract) when no text is found on a page
   - Estimates font sizes from bounding boxes for proper heading detection

2. **Heading Detection**:
   - Employs k-means clustering to identify heading levels based on font sizes
   - Uses heuristics to filter out footers, page numbers, and other non-heading text
   - Handles multi-line headings and numbered lists intelligently

3. **Performance Optimizations**:
   - Uses lightweight PaddleOCR models to stay within the 200MB limit
   - Implements parallel processing for handling multiple PDFs efficiently
   - Minimizes memory usage through efficient data structures and processing

## Models and Libraries Used

- **PyMuPDF (fitz)**: For PDF text extraction and rendering
- **PaddleOCR**: Primary OCR engine with pre-trained English models
- **Tesseract OCR**: Fallback OCR engine
- **scikit-learn**: For k-means clustering of font sizes
- **Python 3.9**: Base runtime

## Building and Running

### Prerequisites

- Docker
- At least 2GB of free disk space
- 8GB RAM recommended for optimal performance

### Building the Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor:latest .
```

### Running the Container

To process all PDFs in an input directory:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-outline-extractor:latest
```

### Input/Output

- **Input**: Place PDF files in the `./input` directory
- **Output**: JSON files will be generated in the `./output` directory with the same base filename as the input PDF

### Example Output

For each input PDF (e.g., `document.pdf`), a corresponding `document.json` will be created with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "text": "Chapter 1: Introduction",
      "level": 1,
      "page": 0
    },
    {
      "text": "1.1 Background",
      "level": 2,
      "page": 1
    },
    ...
  ]
}
```

## Performance

- **Model Size**: < 200MB (compressed)
- **Processing Time**: < 10 seconds for a 50-page PDF (on 8 CPU cores)
- **Memory Usage**: < 1GB peak

## Constraints and Limitations

- Designed to run on CPU only (no GPU acceleration)
- Optimized for English text (though may work with other languages to some extent)
- Requires PDFs to be properly structured for best results
- May have reduced accuracy with very complex layouts or poor quality scans

## License

This project is proprietary software. All rights reserved.
