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

# Configuracao do logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    Melhora a qualidade da imagem para OCR.
    
    Args:
        image: Imagem no formato numpy array
    
    Returns:
        np.ndarray: Imagem melhorada
    """
    # Converte para PIL Image para ajustes de contraste e brilho
    pil_image = Image.fromarray(image)
    
    # Ajusta contraste
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    
    # Ajusta brilho
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Converte de volta para numpy array
    return np.array(pil_image)

def calculate_dpi(image: np.ndarray) -> float:
    """
    Calcula o DPI aproximado da imagem.
    
    Args:
        image: Imagem no formato numpy array
    
    Returns:
        float: DPI estimado
    """
    height, width = image.shape[:2]
    return min(width, height) / 8.5  # Assume página A4 padrão

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Aplica pré-processamento na imagem para melhorar a extração de texto.
    
    Args:
        image: Imagem no formato numpy array
    
    Returns:
        np.ndarray: Imagem processada
    """
    # Converte para escala de cinza se necessário
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calcula e ajusta DPI
    current_dpi = calculate_dpi(gray)
    if current_dpi < 300:  # Se DPI menor que 300
        scale_factor = 300 / current_dpi
        gray = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # Melhora qualidade da imagem
    enhanced = enhance_image(gray)
    
    # Aplica threshold adaptativo
    binary = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Reduz ruído
    denoised = cv2.fastNlMeansDenoising(binary)
    
    # Aumenta nitidez
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def extract_text_with_easyocr(image: np.ndarray) -> str:
    """
    Extrai texto usando EasyOCR.
    
    Args:
        image: Imagem no formato numpy array
    
    Returns:
        str: Texto extraído
    """
    try:
        reader = easyocr.Reader(['en'], gpu=False)
        # Configurações para melhor precisão
        results = reader.readtext(
            image,
            detail=0,  # Retorna apenas o texto
            paragraph=True,  # Agrupa texto em parágrafos
            contrast_ths=0.3,  # Ajusta contraste
            adjust_contrast=1.5,  # Aumenta contraste
            text_threshold=0.7,  # Limite de confiança para detecção de texto
            width_ths=0.7,  # Limite para largura de caracteres
            height_ths=0.7  # Limite para altura de caracteres
        )
        return ' '.join(results)
    except Exception as e:
        logger.warning(f"Erro ao extrair texto com EasyOCR: {str(e)}")
        return ""

def extract_text_with_tesseract(image: np.ndarray) -> str:
    """
    Extrai texto usando Tesseract.
    
    Args:
        image: Imagem no formato numpy array
    
    Returns:
        str: Texto extraído
    """
    try:
        custom_config = r'--oem 3 --psm 1'
        return pytesseract.image_to_string(image, config=custom_config)
    except Exception as e:
        logger.warning(f"Erro ao extrair texto com Tesseract: {str(e)}")
        return ""

def clean_extracted_text(text: str) -> str:
    """
    Limpa e formata o texto extraído.
    
    Args:
        text: Texto extraído do OCR
    
    Returns:
        str: Texto limpo e formatado
    """
    # Remove caracteres especiais mantendo pontuação importante
    text = re.sub(r'[^\w\s\-:@.,()"]', ' ', text)
    
    # Normaliza espaços e quebras de linha
    text = re.sub(r'\s+', ' ', text)
    text = text.replace(' :', ':').replace(' .', '.')
    
    # Corrige problemas comuns de OCR
    text = text.replace('0nal', 'onal')
    text = text.replace('W ed g e', 'KNOWLEDGE')
    text = text.replace('SCRUMFOUNDAuloN', 'SCRUM FOUNDATION')
    text = text.replace('DRECTO R', 'DIRECTOR')
    text = text.replace('otESSIONA', 'PROFESSIONAL')
    
    # Remove linhas vazias ou só com espaços
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Remove linhas duplicadas mantendo a ordem
    seen = set()
    cleaned_lines = []
    for line in lines:
        normalized_line = ' '.join(line.split())  # Normaliza espaços internos
        if normalized_line not in seen:
            seen.add(normalized_line)
            cleaned_lines.append(normalized_line)
    
    return '\n'.join(cleaned_lines)

def extract_text_from_image(image_path: str) -> str:
    """
    Extrai texto de uma imagem usando múltiplos OCRs em cascata.
    
    Args:
        image_path: Caminho da imagem
    
    Returns:
        str: Texto extraído
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Aplica pré-processamento
    processed = preprocess_image(image)
    
    # Salva imagem processada para debug
    debug_path = "imagem_processada.png"
    cv2.imwrite(debug_path, processed)
    logger.info(f"Imagem processada salva em: {debug_path}")
    
    # Tenta extrair texto com EasyOCR primeiro
    text = extract_text_with_easyocr(processed)
    if text.strip():
        text = clean_extracted_text(text)
        if len(text) > 100:  # Se encontrou texto suficiente
            logger.info("Texto extraído com sucesso usando EasyOCR")
            return text
    
    # Se EasyOCR falhou, usa Tesseract como última opção
    text = extract_text_with_tesseract(processed)
    if text.strip():
        text = clean_extracted_text(text)
        logger.info("Texto extraído com sucesso usando Tesseract")
        return text
    
    logger.warning("Nenhum OCR conseguiu extrair texto suficiente")
    return text

def extract_text_from_pdf(pdf_path: str, start_page: int = 0, max_pages: Optional[int] = None) -> str:
    """
    Extrai texto do PDF usando OCR.
    
    Args:
        pdf_path: Caminho do arquivo PDF
        start_page: Página inicial (começando de 0)
        max_pages: Número máximo de páginas a processar
    
    Returns:
        str: Texto extraído do PDF
    """
    all_texts = []
    
    try:
        # Extrai texto usando OCR
        logger.info("Convertendo PDF para imagens...")
        with tempfile.TemporaryDirectory() as temp_dir:
            # Converte PDF para imagens
            images = convert_from_path(pdf_path)
            
            for i, image in enumerate(images[start_page:]):
                if max_pages and i >= max_pages:
                    break
                
                # Salva a imagem temporariamente
                image_path = os.path.join(temp_dir, f'page_{i}.png')
                image.save(image_path, 'PNG')
                
                # Extrai texto da imagem
                logger.info(f"Extraindo texto da página {i} usando OCR...")
                text = extract_text_from_image(image_path)
                if text.strip():
                    all_texts.append(text)
        
        # Combina todos os textos extraídos
        combined_text = "\n".join(all_texts)
        
        # Limpa o texto
        combined_text = re.sub(r'\n\s*\n', '\n\n', combined_text)
        combined_text = re.sub(r' +', ' ', combined_text)
        combined_text = combined_text.strip()
        
        # Remove caracteres não imprimíveis
        combined_text = ''.join(char for char in combined_text if char.isprintable() or char == '\n')
        
        # Remove linhas duplicadas mantendo a ordem
        seen = set()
        cleaned_lines = []
        for line in combined_text.split('\n'):
            line = line.strip()
            if line and line not in seen:
                seen.add(line)
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    except Exception as e:
        logger.error(f"Erro ao extrair texto do PDF: {str(e)}")
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

def main():
    """Função principal para executar a extração e exibir resultados."""
    pdf_path = "teste.pdf"
    classifier = TextClassifier()

    try:
        # Extrai texto do PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Extrai informações do certificado
        info = extract_certificate_data(text, classifier)
        
        # Exibe os resultados
        print("\nInformações do Certificado:")
        print("-" * 50)
        
        # Campos principais
        campos_principais = {
            "Nome": info['nome'],
            "Tipo de Certificado": info['tipo_certificado'],
            "Data": info['data'],
            "ID CP": info['id_cp']
        }
        
        for campo, valor in campos_principais.items():
            if valor:
                print(f"{campo}: {valor}")
        
        # Informações adicionais
        campos_adicionais = {
            "Texto Adicional": info['texto_lateral'],
            "Rodapé": info['rodape'],
            "Assinatura": info['assinatura']
        }
        
        campos_preenchidos = {k: v for k, v in campos_adicionais.items() if v}
        if campos_preenchidos:
            print("\nInformações Adicionais:")
            for campo, valor in campos_preenchidos.items():
                print(f"{campo}: {valor}")
        
        print("-" * 50)
        print("\nArquivos gerados:")
        print("- texto_extraido.txt (texto completo extraído do certificado)")
        print("- imagem_processada.png (imagem após pré-processamento)")
        
        # Se o usuário quiser adicionar exemplos de treinamento
        print("\nDeseja adicionar exemplos de treinamento para melhorar a classificação? (s/n)")
        resposta = input().lower()
        if resposta == 's':
            print("\nPara cada campo, digite o texto que representa corretamente:")
            for campo in ["nome", "data", "id", "tipo_documento"]:
                print(f"\nExemplo de {campo}:")
                exemplo = input().strip()
                if exemplo:
                    classifier.add_training_example(exemplo, campo)
                    print(f"Exemplo de {campo} adicionado com sucesso!")
        
    except Exception as e:
        print(f"\nErro: {str(e)}")

if __name__ == "__main__":
    main()
