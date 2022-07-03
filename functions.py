import os
import json
import io
import PyPDF2
from openpyxl import load_workbook
# import pytesseract
# import pandas as pd
# import streamlit as st
import numpy as np
from xml.etree import ElementTree
import easyocr as ocr
from PIL import Image

def file_to_text(file):

    extention = file.name.split('.')[-1]
    file_as_text = ""

    if "csv" in extention  :
        file_as_text = csv_to_text(file=file)
    elif "xlsx" in extention or "xls" in extention:
        file_as_text = xls_to_csv(file=file)
    elif "pdf" in extention:
        file_as_text = pdf_to_text(file=file)
    elif "txt" in extention:
        file_as_text = txt_to_text(file=file)
    elif "xml" in extention:
        file_as_text = xml_to_text(file=file)
    elif "jpg" in extention or  "jpeg" in extention :
        file_as_text = image_to_text(file=file)
    # elif "svg" in extention or "png" in extention or "webp" in extention :
        # only for tesseract ocr
        # file_as_text = image_to_text(file=file)
    else:
        file_as_text = f"🐛 : Not a supported type of file : { extention } "

    return file_as_text

def csv_to_text(file):

    return file.getvalue().decode("utf-8")

def txt_to_text(file):

    return file.getvalue().decode("utf-8")

def xml_to_text(file):

    # in case the file was read before, we set the reader to the begining
    file.seek(0) 
    xml = ElementTree.fromstring( file.read() )
    xml_str = ElementTree.tostring(xml, encoding='unicode')

    return xml_str

def pdf_to_text(file):
    pdfR=PyPDF2.PdfFileReader(file)
    text_of_pdf = ""

    for page in pdfR.pages:
        text_of_pdf += page.extract_text() + "\n"

    # alternative code with fitz  : 
    # return text_of_pdf
    # import fitz
    # doc = fitz.open("pdf", file)
    # for page in doc:
    #     text_of_pdf += page.get_text() + "\n"

    return text_of_pdf

def files_to_text(files):

    def file_separator(number_of_file,name_of_file):
        # No name for extraction usecases
        return f"\n\n|-||-| 🐛 : File 📋 N° {number_of_file+1} |-||-|\n\n"
    
    files_as_text = ""
    if len(files) == 0:
        return "\n\n 🐛 : No file 📋 detected ⁉️ \n\n"

    for number_of_file in range(len(files)):
        file = files[number_of_file]
        name_of_file = file.name
        file_separator_text = file_separator(number_of_file=number_of_file,name_of_file=name_of_file)
        files_as_text += file_separator_text + file_to_text(file)
    
    return files_as_text

def xls_to_csv(file):

    bytes_in = io.BytesIO(file.read())
    wb = load_workbook(bytes_in)
    text_of_xls = ""

    for sheet in wb:
        text_of_xls += sheet.title + "\n\n"
        for row in sheet.rows:
            for cell in row:
                text_of_xls += str(cell.value) + " "
            text_of_xls += "\n"
        
    return text_of_xls

def image_to_text(file):

    # alternative code with tesseract ocr  : 
    # contents = file.read()
    # pil_image = Image.open(io.BytesIO(contents))
    # # Simple image to string
    # text_of_image = pytesseract.image_to_string(pil_image)

    contents = file.read()
    pil_image = Image.open(io.BytesIO(contents))

    reader = ocr.Reader(["en"], model_storage_directory=".")
    result = reader.readtext(
        np.array(pil_image)
    )
    text_of_image = " "
    for text in result:
        text_of_image += text[1] + "\n "

    return text_of_image