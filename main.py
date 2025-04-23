#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
from datetime import datetime
import logging
import os
from typing import Dict, Optional, List, Tuple
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import tempfile
import easyocr
from PIL import Image, ImageEnhance
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import layoutparser as lp

# Configuracao do logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configurar o caminho do Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configurar o caminho do Poppler
os.environ['PATH'] = r'C:\Program Files\poppler\poppler-24.08.0\Library\bin' + os.pathsep + os.environ['PATH']

# Diretório para armazenar dados de treinamento
TRAINING_DATA_DIR = Path("training_data")
TRAINING_DATA_DIR.mkdir(exist_ok=True)

class TextClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])
        self.training_data = self._load_training_data()
        self._train_model()

    def _load_training_data(self) -> Dict[str, List[str]]:
        """Carrega dados de treinamento do diretório."""
        data = {
            "nome": [],
            "data": [],
            "id": [],
            "tipo_documento": [],
            "outros": []
        }
        
        training_file = TRAINING_DATA_DIR / "training_data.json"
        if training_file.exists():
            with open(training_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                data.update(saved_data)
        
        return data

    def _save_training_data(self):
        """Salva dados de treinamento no diretório."""
        training_file = TRAINING_DATA_DIR / "training_data.json"
        with open(training_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)

    def _train_model(self):
        """Treina o modelo com os dados disponíveis."""
        if not any(self.training_data.values()):
            return

        X = []
        y = []
        for category, texts in self.training_data.items():
            X.extend(texts)
            y.extend([category] * len(texts))

        if X and y:
            self.model.fit(X, y)
            self._save_model()

    def _save_model(self):
        """Salva o modelo treinado."""
        model_file = TRAINING_DATA_DIR / "classifier_model.joblib"
        joblib.dump(self.model, model_file)

    def _load_model(self):
        """Carrega o modelo treinado."""
        model_file = TRAINING_DATA_DIR / "classifier_model.joblib"
        if model_file.exists():
            self.model = joblib.load(model_file)

    def classify_text(self, text: str) -> str:
        """Classifica um texto em uma das categorias."""
        if not any(self.training_data.values()):
            return "outros"
        
        try:
            return self.model.predict([text])[0]
        except:
            return "outros"

    def add_training_example(self, text: str, category: str):
        """Adiciona um novo exemplo de treinamento."""
        if category not in self.training_data:
            logger.warning(f"Categoria {category} não reconhecida")
            return

        self.training_data[category].append(text)
        self._save_training_data()
        self._train_model()

    def get_suggested_categories(self, text: str) -> List[Tuple[str, float]]:
        """Retorna categorias sugeridas com suas probabilidades."""
        if not any(self.training_data.values()):
            return [("outros", 1.0)]
        
        try:
            probs = self.model.predict_proba([text])[0]
            categories = self.model.classes_
            return list(zip(categories, probs))
        except:
            return [("outros", 1.0)]

def detect_layout(image: np.ndarray) -> List[Dict]:
    """
    Detecta o layout do documento usando LayoutParser
    """
    # Inicializar modelo de layout
    model = lp.Detectron2LayoutModel(
        config_path='lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
    )
    
    # Detectar layout
    layout = model.detect(image)
    
    # Ordenar blocos por posição vertical e depois horizontal
    layout = lp.Layout([b for b in layout])
    layout = layout.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))
    
    return layout

def extract_text_with_layout(image: np.ndarray, layout: List[Dict], reader) -> Tuple[str, List[Dict]]:
    """
    Extrai texto preservando o layout
    """
    extracted_text = []
    structured_data = []
    
    # Agrupar blocos por linhas baseado na posição vertical
    y_tolerance = 10  # pixels de tolerância para considerar mesma linha
    lines = {}
    
    # Encontrar a largura máxima do documento
    height, width = image.shape[:2]
    
    for block in layout:
        y1 = block.coordinates[1]
        line_found = False
        
        for line_y in lines.keys():
            if abs(y1 - line_y) <= y_tolerance:
                lines[line_y].append(block)
                line_found = True
                break
        
        if not line_found:
            lines[y1] = [block]
    
    # Processar cada linha
    last_y = 0
    last_was_separator = False
    
    for y, line_blocks in sorted(lines.items()):
        # Adicionar linhas em branco entre blocos distantes
        if last_y > 0 and y - last_y > 3 * y_tolerance and not last_was_separator:
            extracted_text.append("")
        last_y = y

        # Ordenar blocos da linha da esquerda para direita
        line_blocks.sort(key=lambda b: b.coordinates[0])
        
        # Processar blocos de texto na linha
        line_parts = []
        current_x = 0
        
        for block in line_blocks:
            x1, y1, x2, y2 = block.coordinates
            
            # Adicionar espaçamento baseado na posição horizontal
            if x1 > current_x:
                padding = " " * ((x1 - current_x) // 10)
                if padding:
                    line_parts.append(padding)
            
            # Extrair texto da região
            region = image[int(y1):int(y2), int(x1):int(x2)]
            results = reader.readtext(region)
            
            block_text = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if text:
                    block_text.append(text)
                    structured_data.append({
                        'text': text,
                        'bbox': [[x + x1, y + y1] for [x, y] in bbox],
                        'confidence': conf,
                        'type': block.type
                    })
            
            if block_text:
                # Formatar texto baseado no tipo do bloco
                if block.type == "Title":
                    text = "  ".join(block_text).upper()
                elif block.type == "List":
                    text = "• " + " ".join(block_text)
                elif block.type == "Table":
                    text = " | ".join(block_text)
                else:
                    text = " ".join(block_text)
                
                line_parts.append(text)
                current_x = x2
        
        if line_parts:
            # Juntar partes da linha preservando o espaçamento
            line_content = "".join(line_parts).rstrip()
            if line_content:
                extracted_text.append(line_content)
    
    return '\n'.join(extracted_text), structured_data

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Melhora a qualidade da imagem para melhor OCR
    """
    # Converter para escala de cinza se necessário
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Aplicar threshold adaptativo
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Reduzir ruído
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Aumentar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    return enhanced

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    return obj

def process_document(file_path: str) -> Dict:
    """
    Process document and extract text with layout preservation
    """
    try:
        logger.info("Starting text extraction process")
        
        # Initialize EasyOCR
        reader = easyocr.Reader(['en'])
        
        # Convert PDF to images if needed
        if file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            logger.info(f"Successfully converted PDF to {len(images)} images")
        else:
            images = [Image.open(file_path)]
        
        all_text = []
        all_blocks = []
        
        # Process each page
        for i, image in enumerate(images):
            logger.info(f"Processing page {i}")
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Enhance image
            enhanced = enhance_image(image_np)
            
            # Save processed image for debugging
            cv2.imwrite('imagem_processada.png', enhanced)
            
            # Detect layout
            layout = detect_layout(enhanced)
            
            # Extract text with layout
            text, blocks_data = extract_text_with_layout(enhanced, layout, reader)
            
            if text.strip():
                all_text.append(text)
                all_blocks.extend(blocks_data)
        
        # Convert data to serializable format
        result_data = {
            'text': '\n\n=== Page Break ===\n\n'.join(all_text),
            'blocks': all_blocks,
            'pages': len(images)
        }
        
        # Save results
        with open('texto_extraido.txt', 'w', encoding='utf-8') as f:
            f.write(result_data['text'])
        
        with open('texto_extraido_estruturado.json', 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        
        logger.info("Text extraction completed successfully")
        return result_data
        
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def extract_certificate_data(text: str, classifier: Optional[TextClassifier] = None) -> Dict[str, str]:
    """
    Extrai dados específicos do certificado a partir do texto.
    
    Args:
        text: Texto extraído do certificado
        classifier: Classificador de texto opcional
    
    Returns:
        Dict[str, str]: Dicionário com os dados extraídos
    """
    # Salva o texto extraído para debug
    with open("texto_extraido.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    logger.info("Iniciando extração de dados do certificado...")
    
    # Inicializa o dicionário de resultados
    data = {
        "nome": "",
        "tipo_certificado": "",
        "data": "",
        "id_cp": "",
        "texto_lateral": "",
        "rodape": "",
        "assinatura": ""
    }
    
    # Se um classificador for fornecido, use-o para ajudar na extração
    if classifier:
        # Divide o texto em linhas e classifica cada uma
        lines = text.split('\n')
        classified_lines = [(line, classifier.classify_text(line)) for line in lines if line.strip()]
        
        # Agrupa linhas por categoria
        categorized_lines = {}
        for line, category in classified_lines:
            if category not in categorized_lines:
                categorized_lines[category] = []
            categorized_lines[category].append(line)
        
        # Usa as linhas classificadas para ajudar na extração
        for category, lines in categorized_lines.items():
            if category == "nome":
                data["nome"] = " ".join(lines)
            elif category == "data":
                data["data"] = " ".join(lines)
            elif category == "id":
                data["id_cp"] = " ".join(lines)
            elif category == "tipo_documento":
                data["tipo_certificado"] = " ".join(lines)
    
    # Padrões de regex específicos para o formato do certificado SFPC
    patterns = {
        "nome": [
            r"certify that:?\s*([A-Za-z\s]+?)(?=\s+[Hh]as successfully)",
            r"This is to certify that:?\s*([A-Za-z\s]+?)(?=\s+[Hh]as successfully)",
            r"(?i)Nome:?\s*([^\n]+)"
        ],
        "texto_lateral": [
            r"([Hh]as successfully passed the certification exam for\s*SCRUM FOUNDATION PROFESSIONAL CERTIFICATE)",
            r"(SCRUM\s+FOUNDATION\s+PROFESSIONAL\s+CERTIFICATE\s*(?:\(SFPC\))?)"
        ],
        "data": [
            r"Date of Certification:?\s*([^\n]+)",
            r"(?:Tue|Mon|Wed|Thu|Fri|Sat|Sun)?[,\s]*(\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})",
            r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})"
        ],
        "id_cp": [
            r"ID CP:?\s*([A-Z0-9-]+)",
            r"(?<![\w-])([A-Z0-9]+-[A-Z0-9]+-[A-Z0-9]+)(?![\w-])",
            r"Certificate ID:?\s*([A-Z0-9-]+)"
        ],
        "rodape": [
            r"(CertiProf.*?(?:registered trademark|United States).*?(?:countries|LLC))",
            r"(Copyright.*?CertiProf)",
            r"(All rights reserved)"
        ],
        "assinatura": [
            r"(MANAGING\s+DIRECTOR)",
            r"(MA\s*NA\s*GIN\s*G\s*D[IR]RECTOR)",
            r"(Director.*?Signature)",
            r"(Authorized\s+Signatory)"
        ]
    }
    
    # Tenta extrair cada campo usando múltiplos padrões
    for field, field_patterns in patterns.items():
        for pattern in field_patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL | re.IGNORECASE)
            if match:
                if match.groups():
                    extracted_value = match.group(1).strip()
                else:
                    extracted_value = match.group(0).strip()
                
                # Processamento específico para datas
                if field == "data" and extracted_value:
                    try:
                        # Remove o dia da semana e o sufixo th/st/nd/rd
                        if any(day in extracted_value for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']):
                            extracted_value = extracted_value.split(' ', 1)[1]
                        date_str = re.sub(r'(?:st|nd|rd|th)', '', extracted_value)
                        # Tenta diferentes formatos de data
                        for date_format in ["%d %b %Y", "%d %B %Y", "%B %d %Y", "%b %d %Y"]:
                            try:
                                date_obj = datetime.strptime(date_str.strip(), date_format)
                                extracted_value = date_obj.strftime("%d/%m/%Y")
                                break
                            except ValueError:
                                continue
                    except Exception as e:
                        logger.warning(f"Erro ao converter data: {e}")
                        continue
                
                data[field] = extracted_value
                logger.info(f"Campo {field} encontrado: {data[field]}")
                break
        
        if not data[field] and field != "tipo_certificado":
            logger.warning(f"Campo {field} não encontrado no texto")
    
    return data

def group_text_blocks(blocks, max_y_diff=15, max_x_diff=100):
    """Group text blocks into lines and columns based on their positions."""
    if not blocks:
        return []
        
    # Sort blocks by y-coordinate (vertically)
    blocks.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
    
    lines = []
    current_line = []
    current_y = blocks[0][0][0][1]
    
    # Group blocks into lines
    for block in blocks:
        y = block[0][0][1]
        if abs(y - current_y) > max_y_diff:
            if current_line:
                # Sort blocks in line by x-coordinate
                current_line.sort(key=lambda x: x[0][0][0])
                lines.append(current_line)
            current_line = [block]
            current_y = y
        else:
            current_line.append(block)
    
    # Add the last line
    if current_line:
        current_line.sort(key=lambda x: x[0][0][0])
        lines.append(current_line)
    
    # Format lines with proper spacing
    formatted_lines = []
    for line in lines:
        formatted_line = ""
        last_x_end = 0
        
        for block in line:
            x_start = block[0][0][0]
            text = block[1]
            
            # Add spacing based on x-coordinate difference
            if last_x_end > 0:
                x_diff = x_start - last_x_end
                if x_diff > max_x_diff:
                    formatted_line += "    "  # Tab for column separation
                elif x_diff > 20:
                    formatted_line += "  "  # Double space for word separation
                else:
                    formatted_line += " "  # Normal word spacing
            
            formatted_line += text
            last_x_end = block[0][2][0]  # End x-coordinate of current block
        
        # Skip empty lines and lines with just spaces
        if formatted_line.strip():
            formatted_lines.append(formatted_line)
    
    return formatted_lines

def extract_text_with_easyocr(image_path, reader):
    """Extract text from image using EasyOCR with improved layout preservation."""
    logger.info(f"Extracting text from {image_path}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Extract text with bounding boxes
    results = reader.readtext(image)
    
    # Convert numpy types to Python native types
    results = convert_numpy_types(results)
    
    # Group text blocks into lines with proper formatting
    formatted_lines = group_text_blocks(results)
    
    # Combine lines into final text with proper spacing
    text = "\n".join(formatted_lines)
    
    # Add page separator
    text += "\n\n" + "="*80 + "\n\n"
    
    return text, results

def main():
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'])
    
    try:
        # Try to process PDF first
        pdf_path = 'teste.pdf'
        if os.path.exists(pdf_path):
            logger.info("Converting PDF to images...")
            images = convert_from_path(pdf_path)
            
            logger.info(f"Successfully converted PDF to {len(images)} images")
            
            all_text = ""
            all_blocks = []
            
            # Save pages as images and process them
            for i, image in enumerate(images):
                temp_path = f'temp_page_{i}.png'
                image.save(temp_path)
                
                text, blocks = extract_text_with_easyocr(temp_path, reader)
                all_text += f"\n--- Page {i+1} ---\n" + text
                all_blocks.extend(blocks)
                
                os.remove(temp_path)
                
            # Save extracted text
            with open('texto_extraido.txt', 'w', encoding='utf-8') as f:
                f.write(all_text)
                
            # Save block data (already converted to native Python types)
            with open('blocos.json', 'w', encoding='utf-8') as f:
                json.dump(all_blocks, f, ensure_ascii=False, indent=2)
                
            logger.info("Successfully processed PDF")
            
        else:
            # Try to process single image
            image_path = 'imagem_processada.png'
            if os.path.exists(image_path):
                text, blocks = extract_text_with_easyocr(image_path, reader)
                
                with open('texto_extraido.txt', 'w', encoding='utf-8') as f:
                    f.write(text)
                    
                with open('blocos.json', 'w', encoding='utf-8') as f:
                    json.dump(blocks, f, ensure_ascii=False, indent=2)
                    
                logger.info("Successfully processed image")
            else:
                logger.error("No valid input file found")
                return
                
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

if __name__ == "__main__":
    main()
