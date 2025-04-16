# Extrator de Certificados

Este projeto é um extrator de informações de certificados PDF usando OCR (Reconhecimento Óptico de Caracteres). O script é capaz de extrair informações importantes de certificados, como nome, tipo de certificado, data e ID.

## Funcionalidades

- Conversão de PDF para imagem
- Pré-processamento de imagem para melhorar a qualidade do OCR
- Extração de texto usando múltiplos motores OCR (EasyOCR e Tesseract)
- Limpeza e formatação do texto extraído
- Extração estruturada de informações específicas do certificado

## Requisitos

- Python 3.8+
- Tesseract OCR
- Poppler (para conversão de PDF)
- Bibliotecas Python listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone [URL_DO_REPOSITORIO]
cd [NOME_DO_REPOSITORIO]
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Instale o Tesseract OCR:
- Windows: Baixe o instalador em https://github.com/UB-Mannheim/tesseract/wiki
- Linux: `sudo apt-get install tesseract-ocr`
- macOS: `brew install tesseract`

4. Instale o Poppler:
- Windows: Baixe o instalador em https://github.com/oschwartz10612/poppler-windows
- Linux: `sudo apt-get install poppler-utils`
- macOS: `brew install poppler`

## Uso

1. Coloque o arquivo PDF do certificado na pasta do projeto
2. Execute o script:
```bash
python main.py
```

3. O script irá gerar:
- `texto_extraido.txt`: Texto completo extraído do certificado
- `imagem_processada.png`: Imagem após pré-processamento
- Informações extraídas do certificado no console

## Estrutura do Projeto

- `main.py`: Script principal com todas as funções de extração
- `requirements.txt`: Lista de dependências do projeto
- `.gitignore`: Arquivo para controle de versão

## Contribuição

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.

## Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para mais detalhes. 