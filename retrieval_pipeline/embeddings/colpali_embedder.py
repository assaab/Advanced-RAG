"""
ColPali Document Page Embedder
Vision-Language model for embedding PDF pages as patch sequences
"""
import asyncio
import base64
import io
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

# Core dependencies
import numpy as np
from PIL import Image

# HuggingFace transformers
try:
    from transformers import AutoProcessor, AutoModel
    import torch
    import torch.nn.functional as F
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# PDF processing
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    import pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ColPaliConfig:
    """Configuration for ColPali embeddings"""
    model_name: str = "vidore/colpali-v1.2"  # Official ColPali model
    device: Optional[str] = None
    torch_dtype = torch.float16
    trust_remote_code: bool = True
    
    # Image processing
    max_image_size: Tuple[int, int] = (1024, 1024)
    patch_size: int = 16  # Standard patch size for vision transformers
    dpi: int = 150  # DPI for PDF to image conversion
    
    # Embedding settings
    embedding_dim: int = 128  # ColPali embedding dimension
    max_patches_per_page: int = 1030  # Typical for ColPali
    
    # Processing settings
    batch_size: int = 4
    enable_caching: bool = True
    cache_dir: Optional[str] = None


@dataclass  
class ColPaliPageEmbedding:
    """Single page embedding result from ColPali"""
    page_number: int
    patch_embeddings: List[List[float]]  # List of patch embeddings (1030 × 128)
    image_size: Tuple[int, int]
    patch_count: int
    metadata: Dict[str, Any]


@dataclass
class ColPaliDocumentEmbedding:
    """Complete document embedding with all pages"""
    document_id: str
    page_embeddings: List[ColPaliPageEmbedding]
    model_name: str
    total_patches: int
    processing_time_seconds: float


class ColPaliEmbedder:
    """
    ColPali embedder for visual document understanding
    
    ColPali embeds PDF pages as sequences of visual patches, enabling:
    - Visual document search
    - Layout-aware retrieval  
    - Multi-modal query understanding
    - Fine-grained visual matching
    """
    
    def __init__(self, config: Optional[ColPaliConfig] = None):
        self.config = config or ColPaliConfig()
        
        if not HF_AVAILABLE:
            raise ImportError("transformers and torch required for ColPali")
        
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model components
        self.processor = None
        self.model = None
        
        # Load model
        self._load_model()
        
        # Cache for embeddings
        self._cache = {} if self.config.enable_caching else None
        
        logger.info(f"ColPali embedder initialized on {self.device}")
    
    def _load_model(self):
        """Load ColPali model and processor"""
        try:
            logger.info(f"Loading ColPali model: {self.config.model_name}")
            
            # Load processor (handles image preprocessing)
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Load model
            self.model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=self.config.trust_remote_code
            )
            
            self.model.eval()
            logger.info(f"ColPali model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ColPali model: {e}")
            # Fallback to mock implementation for development
            self._load_mock_model()
    
    def _load_mock_model(self):
        """Load mock model for development/testing"""
        logger.warning("Using mock ColPali model for development")
        
        class MockProcessor:
            def __call__(self, images, **kwargs):
                # Mock processor that returns dummy tensors
                batch_size = len(images) if isinstance(images, list) else 1
                return {
                    "pixel_values": torch.randn(batch_size, 3, 224, 224),
                    "input_ids": torch.randint(0, 1000, (batch_size, 10))
                }
        
        class MockModel:
            def __call__(self, **inputs):
                batch_size = inputs["pixel_values"].shape[0] 
                # Mock output with patch embeddings
                return type('MockOutput', (), {
                    'last_hidden_state': torch.randn(batch_size, self.config.max_patches_per_page, self.config.embedding_dim)
                })()
            
            def eval(self):
                pass
        
        self.processor = MockProcessor()
        self.model = MockModel()
    
    async def embed_pdf_pages(
        self, 
        pdf_path: Union[str, Path],
        page_range: Optional[Tuple[int, int]] = None
    ) -> ColPaliDocumentEmbedding:
        """
        Embed all pages of a PDF document using ColPali
        
        Args:
            pdf_path: Path to PDF file
            page_range: Optional (start, end) page range (0-indexed)
        
        Returns:
            ColPaliDocumentEmbedding with all page embeddings
        """
        import time
        start_time = time.time()
        
        pdf_path = Path(pdf_path)
        document_id = pdf_path.stem
        
        logger.info(f"Starting ColPali embedding for {pdf_path}")
        
        # Convert PDF pages to images
        page_images = await self._pdf_to_images(pdf_path, page_range)
        
        if not page_images:
            raise ValueError(f"No pages extracted from {pdf_path}")
        
        # Embed pages in batches
        page_embeddings = []
        
        for i in range(0, len(page_images), self.config.batch_size):
            batch_images = page_images[i:i + self.config.batch_size]
            batch_page_numbers = list(range(i, min(i + self.config.batch_size, len(page_images))))
            
            batch_embeddings = await self._embed_image_batch(batch_images, batch_page_numbers)
            page_embeddings.extend(batch_embeddings)
        
        processing_time = time.time() - start_time
        total_patches = sum(emb.patch_count for emb in page_embeddings)
        
        result = ColPaliDocumentEmbedding(
            document_id=document_id,
            page_embeddings=page_embeddings,
            model_name=self.config.model_name,
            total_patches=total_patches,
            processing_time_seconds=processing_time
        )
        
        logger.info(f"ColPali embedding completed: {len(page_embeddings)} pages, {total_patches} patches in {processing_time:.2f}s")
        return result
    
    async def embed_images(
        self, 
        images: List[Union[Image.Image, str, Path]],
        image_ids: Optional[List[str]] = None
    ) -> List[ColPaliPageEmbedding]:
        """
        Embed a list of images using ColPali
        
        Args:
            images: List of PIL Images, file paths, or base64 strings
            image_ids: Optional list of image identifiers
        
        Returns:
            List of ColPaliPageEmbedding objects
        """
        if image_ids and len(image_ids) != len(images):
            raise ValueError("image_ids length must match images length")
        
        # Load and prepare images
        pil_images = []
        for i, img in enumerate(images):
            if isinstance(img, (str, Path)):
                if str(img).startswith('data:image') or len(str(img)) > 1000:
                    # Base64 encoded image
                    pil_img = self._decode_base64_image(str(img))
                else:
                    # File path
                    pil_img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                pil_img = img.convert("RGB")
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
            
            # Resize if needed
            if max(pil_img.size) > max(self.config.max_image_size):
                pil_img.thumbnail(self.config.max_image_size, Image.Resampling.LANCZOS)
            
            pil_images.append(pil_img)
        
        # Process in batches
        embeddings = []
        for i in range(0, len(pil_images), self.config.batch_size):
            batch_images = pil_images[i:i + self.config.batch_size]
            batch_numbers = list(range(i, min(i + self.config.batch_size, len(pil_images))))
            
            batch_embeddings = await self._embed_image_batch(batch_images, batch_numbers)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    async def _pdf_to_images(
        self, 
        pdf_path: Path, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Image.Image]:
        """Convert PDF pages to PIL Images"""
        
        # Try PyMuPDF first (faster)
        if FITZ_AVAILABLE:
            try:
                return await self._pdf_to_images_fitz(pdf_path, page_range)
            except Exception as e:
                logger.warning(f"PyMuPDF failed, trying pdf2image: {e}")
        
        # Fallback to pdf2image
        if PDF2IMAGE_AVAILABLE:
            try:
                return await self._pdf_to_images_pdf2image(pdf_path, page_range)
            except Exception as e:
                logger.error(f"pdf2image failed: {e}")
                raise
        
        raise RuntimeError("No PDF to image converter available. Install PyMuPDF or pdf2image.")
    
    async def _pdf_to_images_fitz(
        self, 
        pdf_path: Path, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Image.Image]:
        """Convert PDF to images using PyMuPDF (fitz)"""
        
        doc = fitz.open(pdf_path)
        images = []
        
        start_page = page_range[0] if page_range else 0
        end_page = page_range[1] if page_range else doc.page_count
        end_page = min(end_page, doc.page_count)
        
        for page_num in range(start_page, end_page):
            page = doc[page_num]
            
            # Render page as image
            mat = fitz.Matrix(self.config.dpi / 72, self.config.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("ppm")
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(img)
        
        doc.close()
        return images
    
    async def _pdf_to_images_pdf2image(
        self, 
        pdf_path: Path, 
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[Image.Image]:
        """Convert PDF to images using pdf2image"""
        
        first_page = (page_range[0] + 1) if page_range else None  # pdf2image uses 1-based indexing
        last_page = (page_range[1]) if page_range else None
        
        images = pdf2image.convert_from_path(
            pdf_path,
            dpi=self.config.dpi,
            first_page=first_page,
            last_page=last_page,
            fmt='RGB'
        )
        
        return images
    
    async def _embed_image_batch(
        self, 
        images: List[Image.Image], 
        page_numbers: List[int]
    ) -> List[ColPaliPageEmbedding]:
        """Embed a batch of images using ColPali"""
        
        with torch.no_grad():
            try:
                # Process images
                inputs = self.processor(
                    images=images,
                    return_tensors="pt",
                    do_rescale=True,
                    do_normalize=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs)
                
                # Extract embeddings (typically last_hidden_state)
                if hasattr(outputs, 'last_hidden_state'):
                    embeddings_tensor = outputs.last_hidden_state
                elif hasattr(outputs, 'pooler_output'):
                    embeddings_tensor = outputs.pooler_output
                else:
                    # Use the first tensor-like output
                    embeddings_tensor = next(v for v in outputs.__dict__.values() 
                                           if isinstance(v, torch.Tensor))
                
                # Process embeddings for each image
                results = []
                
                for i, (image, page_num) in enumerate(zip(images, page_numbers)):
                    # Extract embeddings for this image
                    if len(embeddings_tensor.shape) == 3:  # [batch, patches, dim]
                        image_embeddings = embeddings_tensor[i]
                    else:  # [batch, dim] - pooled output
                        image_embeddings = embeddings_tensor[i].unsqueeze(0)
                    
                    # Normalize embeddings
                    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
                    
                    # Convert to list
                    patch_embeddings = image_embeddings.cpu().numpy().tolist()
                    
                    result = ColPaliPageEmbedding(
                        page_number=page_num,
                        patch_embeddings=patch_embeddings,
                        image_size=image.size,
                        patch_count=len(patch_embeddings),
                        metadata={
                            "original_size": image.size,
                            "model_name": self.config.model_name,
                            "embedding_dim": len(patch_embeddings[0]) if patch_embeddings else 0
                        }
                    )
                    
                    results.append(result)
                
                return results
                
            except Exception as e:
                logger.error(f"Failed to embed image batch: {e}")
                # Return mock embeddings for development
                return self._create_mock_embeddings(images, page_numbers)
    
    def _create_mock_embeddings(
        self, 
        images: List[Image.Image], 
        page_numbers: List[int]
    ) -> List[ColPaliPageEmbedding]:
        """Create mock embeddings for development/testing"""
        
        results = []
        
        for image, page_num in zip(images, page_numbers):
            # Create random patch embeddings
            num_patches = min(self.config.max_patches_per_page, 100)  # Smaller for mock
            patch_embeddings = np.random.normal(0, 1, (num_patches, self.config.embedding_dim)).tolist()
            
            result = ColPaliPageEmbedding(
                page_number=page_num,
                patch_embeddings=patch_embeddings,
                image_size=image.size,
                patch_count=num_patches,
                metadata={
                    "original_size": image.size,
                    "model_name": "mock_colpali",
                    "embedding_dim": self.config.embedding_dim
                }
            )
            
            results.append(result)
        
        return results
    
    def _decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode base64 image string to PIL Image"""
        
        # Handle data URLs
        if base64_string.startswith('data:image'):
            header, data = base64_string.split(',', 1)
        else:
            data = base64_string
        
        # Decode
        image_data = base64.b64decode(data)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        return image
    
    def compute_page_similarity(
        self, 
        query_embedding: List[List[float]], 
        page_embedding: ColPaliPageEmbedding
    ) -> float:
        """
        Compute similarity between query and page using MaxSim
        Similar to ColBERT late interaction
        """
        
        if not query_embedding or not page_embedding.patch_embeddings:
            return 0.0
        
        # Convert to numpy for efficient computation
        query_matrix = np.array(query_embedding)  # [q_patches, dim]
        page_matrix = np.array(page_embedding.patch_embeddings)  # [page_patches, dim]
        
        # Compute similarity matrix
        similarity_matrix = np.dot(query_matrix, page_matrix.T)  # [q_patches, page_patches]
        
        # MaxSim: max similarity for each query patch, then average
        max_similarities = np.max(similarity_matrix, axis=1)  # [q_patches]
        maxsim_score = np.mean(max_similarities)
        
        return float(maxsim_score)
    
    async def embed_query_image(self, query_image: Union[Image.Image, str, Path]) -> List[List[float]]:
        """Embed a query image for search"""
        
        embeddings = await self.embed_images([query_image])
        
        if embeddings:
            return embeddings[0].patch_embeddings
        else:
            return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        
        return {
            "model_name": self.config.model_name,
            "device": self.device,
            "embedding_dim": self.config.embedding_dim,
            "max_patches_per_page": self.config.max_patches_per_page,
            "patch_size": self.config.patch_size,
            "max_image_size": self.config.max_image_size,
            "dpi": self.config.dpi
        }


class ColPaliSearchIndex:
    """Search index for ColPali embeddings with efficient similarity search"""
    
    def __init__(self, config: Optional[ColPaliConfig] = None):
        self.config = config or ColPaliConfig()
        self.embeddings_store: Dict[str, ColPaliDocumentEmbedding] = {}
        self.page_index: Dict[str, List[Tuple[str, int]]] = {}  # page_id -> [(doc_id, page_num)]
    
    def add_document(self, embedding: ColPaliDocumentEmbedding):
        """Add document embedding to index"""
        
        self.embeddings_store[embedding.document_id] = embedding
        
        # Index pages
        for page_emb in embedding.page_embeddings:
            page_id = f"{embedding.document_id}_page_{page_emb.page_number}"
            
            if page_id not in self.page_index:
                self.page_index[page_id] = []
            
            self.page_index[page_id].append((embedding.document_id, page_emb.page_number))
    
    def search_similar_pages(
        self, 
        query_embedding: List[List[float]], 
        top_k: int = 10
    ) -> List[Tuple[str, int, float]]:
        """
        Search for most similar pages using MaxSim
        
        Returns:
            List of (document_id, page_number, similarity_score)
        """
        
        results = []
        
        for doc_id, doc_embedding in self.embeddings_store.items():
            for page_emb in doc_embedding.page_embeddings:
                # Compute similarity using MaxSim
                similarity = self._compute_maxsim(query_embedding, page_emb.patch_embeddings)
                
                results.append((doc_id, page_emb.page_number, similarity))
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def _compute_maxsim(
        self, 
        query_patches: List[List[float]], 
        page_patches: List[List[float]]
    ) -> float:
        """Compute MaxSim similarity between query and page patches"""
        
        if not query_patches or not page_patches:
            return 0.0
        
        query_matrix = np.array(query_patches)
        page_matrix = np.array(page_patches)
        
        # Compute similarity matrix
        similarity_matrix = np.dot(query_matrix, page_matrix.T)
        
        # MaxSim scoring
        max_similarities = np.max(similarity_matrix, axis=1)
        maxsim_score = np.mean(max_similarities)
        
        return float(maxsim_score)
    
    def get_page_embedding(self, document_id: str, page_number: int) -> Optional[ColPaliPageEmbedding]:
        """Get specific page embedding"""
        
        if document_id in self.embeddings_store:
            doc_embedding = self.embeddings_store[document_id]
            
            for page_emb in doc_embedding.page_embeddings:
                if page_emb.page_number == page_number:
                    return page_emb
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        
        total_pages = sum(len(doc.page_embeddings) for doc in self.embeddings_store.values())
        total_patches = sum(doc.total_patches for doc in self.embeddings_store.values())
        
        return {
            "documents": len(self.embeddings_store),
            "pages": total_pages,
            "patches": total_patches,
            "avg_patches_per_page": total_patches / total_pages if total_pages > 0 else 0
        }


# Example usage
async def example_colpali_usage():
    """Example usage of ColPali embedder"""
    
    config = ColPaliConfig(
        model_name="vidore/colpali-v1.2",
        embedding_dim=128,
        max_patches_per_page=1030
    )
    
    embedder = ColPaliEmbedder(config)
    
    # Embed a PDF document
    # pdf_path = Path("sample_document.pdf")
    # doc_embedding = await embedder.embed_pdf_pages(pdf_path)
    
    # Create mock embedding for example
    from PIL import Image
    mock_image = Image.new("RGB", (800, 600), "white")
    doc_embedding = ColPaliDocumentEmbedding(
        document_id="sample_doc",
        page_embeddings=await embedder.embed_images([mock_image]),
        model_name=config.model_name,
        total_patches=100,
        processing_time_seconds=1.0
    )
    
    print(f"Embedded document: {doc_embedding.document_id}")
    print(f"Pages: {len(doc_embedding.page_embeddings)}")
    print(f"Total patches: {doc_embedding.total_patches}")
    
    # Create search index
    index = ColPaliSearchIndex(config)
    index.add_document(doc_embedding)
    
    # Search with query image
    query_embedding = await embedder.embed_query_image(mock_image)
    results = index.search_similar_pages(query_embedding, top_k=5)
    
    print(f"Search results: {len(results)} pages")
    for doc_id, page_num, score in results:
        print(f"  {doc_id} page {page_num}: {score:.4f}")


if __name__ == "__main__":
    asyncio.run(example_colpali_usage())
