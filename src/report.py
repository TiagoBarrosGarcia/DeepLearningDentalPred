from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from SaveIndividualDetection import *
from PredictResNet import *
from reportlab.lib.colors import Color
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import datetime
from SaveIndividualDetection import save_bboxes_from_image
from PredictResNet import load_resnet_model, predict_folder
from datetime import datetime, timedelta

from PIL import Image


from datetime import datetime
from SaveIndividualDetection import save_bboxes_from_image
from PredictResNet import load_resnet_model, predict_folder
import os


def finalDiagnosis(image_path):
    # 1. Detectar dentes e salvar imagens individuais
    bbox_info, yolo_img_path, color_map = save_bboxes_from_image(image_path)

    # 2. Carregar modelo de classificação de posição
    resnetModel_path = 'Model/finalModel.pth'
    train_data_dir = 'Data/TeethPosition'
    image_folder = 'Data/individual_detection'

    model, classes, device, transform = load_resnet_model(resnetModel_path, train_data_dir)

    preds_dict = predict_folder(model, classes, device, transform, image_folder)

    report_data = []
    for info in bbox_info:
        fname = info['filename']
        diagnosis = info['class_name']  # Diagnóstico da YOLO
        bbox = info['bbox']

        # Número do dente
        parts = fname.replace(".png", "").split("_")
        if len(parts) >= 2:
            tooth = parts[1]
        else:
            tooth = "Unknown"

        # Posição anatômica predita pela ResNet
        position = preds_dict.get(fname, "Unknown")

        report_data.append({
            "tooth": tooth,
            "position": position,
            "issue": diagnosis,
            "date": datetime.today().strftime('%Y-%m-%d')
        })

    return report_data, yolo_img_path, color_map



def generate_dental_report(data, yolo_img_path=None, color_map=None, filename="relatorio_dental.pdf"):
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    logo_path = "Data/FCUL_logo.png"
    if os.path.exists(logo_path):
        logo_img = Image.open(logo_path)
        max_logo_width = 100
        max_logo_height = 50
        logo_width, logo_height = logo_img.size

        scale = min(max_logo_width / logo_width, max_logo_height / logo_height, 1)
        disp_logo_width = logo_width * scale
        disp_logo_height = logo_height * scale

        logo_x = width - 30 - disp_logo_width  # 50 pts margem direita
        logo_y = height - 30 - disp_logo_height  # 50 pts margem topo

        c.drawImage(logo_path, logo_x, logo_y, width=disp_logo_width, height=disp_logo_height, mask='auto')


    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, height - 50, "DENTAL HEALTH REPORT")

    # Basic info
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 80, f"Date: {datetime.today().strftime('%Y-%m-%d')}")

    # Posicionamento na mesma linha, ajustando X para Sex e Birthdate
    c.drawString(50, height - 95, f"Name: Alexandre Delgado")
    c.drawString(250, height - 95, f"Sex: Male")
    c.drawString(370, height - 95, f"Birthdate: 2003-08-15")

    # Linha horizontal separando
    line_y = height - 105
    c.setLineWidth(0.5)
    c.line(50, line_y, width - 50, line_y)

    # Dentist abaixo da linha
    c.drawString(50, height - 120, f"Dentist: Dr. Deep Model")

    # Draw YOLO image (if available), scale to fit width margin, keep aspect ratio
    if yolo_img_path and os.path.exists(yolo_img_path):
        max_width = width - 200  # Leave space for legend
        max_height = 200
        img = Image.open(yolo_img_path)
        img_width, img_height = img.size

        scale = min(max_width / img_width, max_height / img_height, 1)
        disp_width = img_width * scale
        disp_height = img_height * scale

        img_x = 50
        img_y = height - 150 - disp_height
        c.drawImage(yolo_img_path, img_x, img_y, width=disp_width, height=disp_height)

        # Draw the legend on the right side of the image
        if color_map:
            legend_x = img_x + disp_width + 20
            legend_y = img_y + disp_height - 10  # top align with image
            c.setFont("Helvetica-Bold", 10)
            c.drawString(legend_x, legend_y, "LEGEND")
            legend_y -= 15

            c.setFont("Helvetica", 9)
            for disease, bgr in color_map.items():
                rgb = (bgr[2], bgr[1], bgr[0])  # BGR → RGB
                color = Color(rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)
                c.setFillColor(color)
                c.rect(legend_x, legend_y - 3, 10, 10, fill=1, stroke=0)
                c.setFillColorRGB(0, 0, 0)
                c.drawString(legend_x + 15, legend_y, disease)
                legend_y -= 15

        table_start_y = img_y - 15
    else:
        table_start_y = height - 120


    header_y = table_start_y - 10  # ajuste conforme seu código

    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, header_y, "1.")
    c.drawString(65, header_y, "EXAMINATION AND TREATMENT RECORD")

    # Espaço extra antes da tabela
    table_start_y = header_y - 10

    # --- Tabela centralizada com linhas ---
    col_widths = [80, 100, 200, 80]  # largura de cada coluna
    table_width = sum(col_widths)
    table_x = (width - table_width) / 2  # esquerda da tabela centralizada

    table_header_y = table_start_y - 20
    font_size = 10
    row_height = 20
    c.setFont("Helvetica-Bold", font_size)
    
    # Cabeçalho
    headers = ["Tooth", "Position", "Diagnosis", "Date"]
    x = table_x
    for i, header in enumerate(headers):
        text_width = c.stringWidth(header, "Helvetica-Bold", font_size)
        col_center = x + col_widths[i] / 2
        text_y = table_header_y + (row_height - font_size) / 2
        c.drawString(col_center - text_width / 2, text_y, header)
        x += col_widths[i]

    # Dados da tabela
    c.setFont("Helvetica", font_size)
    y = table_header_y - row_height
    for entry in data:
        x = table_x
        values = [str(entry["tooth"]), str(entry["position"]), entry["issue"], entry["date"]]
        for i, value in enumerate(values):
            text_width = c.stringWidth(value, "Helvetica", font_size)
            col_center = x + col_widths[i] / 2
            text_y = y + (row_height - font_size) / 2
            c.drawString(col_center - text_width / 2, text_y, value)
            x += col_widths[i]

        # linhas da tabela (horizontal e vertical) — mantém igual
        c.line(table_x, y + row_height, table_x + table_width, y + row_height)
        x = table_x
        for w in col_widths:
            c.line(x, y + row_height, x, y)
            x += w
        c.line(table_x + table_width, y + row_height, table_x + table_width, y)

        y -= row_height

    # Ajusta y para dar espaço para a seção 3 (última linha visível)
    y += row_height  # volta uma linha para garantir que a última linha fique visível

    # Agora y é o ponto de referência para a próxima seção
    section3_y = y - 30
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, section3_y, "3. AT THIS VISIT PATIENT RECEIVED:")

    checkbox_size = 12
    spacing = 110
    labels = ["CLEANING", "TREATMENT", "FLUORIDE", "EXAM UNABLE TO BE DONE"]
    
    # Check logic: treatment if data exists
    treatment_done = len(data) > 0
    checks = [True, treatment_done, False, False]

    x = 50
    for i, (label, checked) in enumerate(zip(labels, checks)):
        box_x = x + (i * spacing)
        box_y = section3_y - 18

        c.setLineWidth(1)
        c.rect(box_x, box_y, checkbox_size, checkbox_size)

        if checked:
            c.setLineWidth(2)
            c.line(box_x + 2, box_y + 5, box_x + 5, box_y + 2)
            c.line(box_x + 5, box_y + 2, box_x + 10, box_y + 10)

        c.setFont("Helvetica", 10)
        c.drawString(box_x + checkbox_size + 5, box_y, label)

    # Section 4 - NEXT STEPS
    section4_y = section3_y - 50
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, section4_y, "4. NEXT STEPS:")

    checkbox_size = 12
    spacing = 200

    labels = ["TREATMENT", "NEXT APPOINTMENT:"]
    checks = [len(data) > 0, len(data) == 0]

    # Acha a posição mínima Y ocupada pelos checkboxes da seção 4
    min_y = section4_y
    for i, (label, checked) in enumerate(zip(labels, checks)):
        box_x = 50 + i * spacing
        box_y = section4_y - 18
        min_y = min(min_y, box_y)

        c.setLineWidth(1)
        c.rect(box_x, box_y, checkbox_size, checkbox_size)

        if checked:
            c.setLineWidth(2)
            c.line(box_x + 2, box_y + 5, box_x + 5, box_y + 2)
            c.line(box_x + 5, box_y + 2, box_x + 10, box_y + 10)

        c.setFont("Helvetica", 10)
        c.drawString(box_x + checkbox_size + 5, box_y, label)

        if label == "NEXT APPOINTMENT:" and checked:
            next_date = (datetime.today() + timedelta(days=180)).strftime('%Y-%m-%d')
            c.drawString(box_x + checkbox_size + 5 + 115, box_y, f"{next_date}")

    # Agora, coloca o SUMMARY um pouco abaixo da parte mais baixa da seção 4 (checkboxes)
    summary_y = min_y - 50  
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, summary_y, "5. SUMMARY:")

    line_spacing = 25
    c.setLineWidth(1)
    c.line(50, summary_y - 20, 550, summary_y - 20)
    c.line(50, summary_y - 20 - line_spacing, 550, summary_y - 20 - line_spacing)

    # Assinatura fica abaixo do summary
    sig_y = summary_y - 80  # distância segura para não sobrepor

    # ---- Section 6 - Signature ----
    c.line(50, sig_y, 250, sig_y)
    c.setFont("Helvetica", 10)
    c.drawString(50, sig_y - 12, "Signature of the Dentist")

    c.line(300, sig_y, 450, sig_y)
    c.setFont("Helvetica", 10)
    c.drawString(300, sig_y - 12, "Date")

    c.save()
    print(f"Report saved as {filename}")


if __name__ == "__main__":
    image_path = 'Data/xray-x-ray-2764828_1280.jpg'
    report_data, yolo_img_path, color_map = finalDiagnosis(image_path)
    generate_dental_report(report_data, yolo_img_path, color_map)
