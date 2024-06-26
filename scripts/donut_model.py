import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, AutoModel, DonutProcessor, VisionEncoderDecoderModel
import re
import torch
import cv2
import numpy as np


class DonutModel:
    def __init__(self):
        self.processor = DonutProcessor.from_pretrained(
            "sccengizlrn/invoices-donut-model-v1")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "sccengizlrn/invoices-donut-model-v1")

    def parse_image(self, image):
        try:
            # Tokenizer ile encoder inputs hazırlama
            pixel_values = self.processor(
                image, return_tensors="pt").pixel_values

            # Tokenizer ile decoder inputs hazırlama
            task_prompt = "<s_cord-v2>"
            decoder_input_ids = self.processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)

            outputs = self.model.generate(pixel_values.to(device),
                                          decoder_input_ids=decoder_input_ids.to(
                                              device),
                                          max_length=self.model.decoder.config.max_position_embeddings,
                                          early_stopping=True,
                                          pad_token_id=self.processor.tokenizer.pad_token_id,
                                          eos_token_id=self.processor.tokenizer.eos_token_id,
                                          use_cache=True,
                                          num_beams=1,
                                          bad_words_ids=[
                                              [self.processor.tokenizer.unk_token_id]],
                                          return_dict_in_generate=True,
                                          output_scores=True,)

            # Model ile cevap üretme
            outputs = self.model.generate(pixel_values.to(device),
                                          decoder_input_ids=decoder_input_ids.to(
                                              device),
                                          max_length=self.model.decoder.config.max_position_embeddings,
                                          early_stopping=True,
                                          pad_token_id=self.processor.tokenizer.pad_token_id,
                                          eos_token_id=self.processor.tokenizer.eos_token_id,
                                          use_cache=True,
                                          num_beams=1,
                                          bad_words_ids=[
                                              [self.processor.tokenizer.unk_token_id]],
                                          return_dict_in_generate=True,
                                          output_scores=True,)

            # Cevabı işleme
            sequence = self.processor.batch_decode(outputs.sequences)[0]
            sequence = sequence.replace(self.processor.tokenizer.eos_token, "").replace(
                self.processor.tokenizer.pad_token, "")
            # ilk görev başlangıç belirteci kaldırma
            sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()

            return self.processor.token2json(sequence)

        except Exception as e:
            print("Hata:", str(e))
            return None

    def preprocess_pdf_image(self, image):
        # Yatay çizgileri tespit et ve ilgili verileri al
        proceed_image_data = self.detect_horizontal_lines(image)
        horizontal_lines_percentages = proceed_image_data["line_percentages"]
        print(
            f"horizontal lines percentages: {str(horizontal_lines_percentages)}")

        # Yatay çizgilerin altında kalan alanları silerek kırpılmış görüntüler oluştur
        cropped_images = [self.erase_below_y(image, horizontal_line_percentage)
                          for horizontal_line_percentage in horizontal_lines_percentages]
        # Orijinal görüntüyü de kırpılmış görüntüler listesine ekle
        cropped_images.append(image)

        # Kırpılmış her bir görüntüyü parse et ve sonuçları al
        results = [self.parse_image(cropped_image)
                   for cropped_image in cropped_images]

        for i, result in enumerate(results):
            print(f"{i}. result: {result}")

        # Sonuçları string'e çevir ve en uzun sonucu bul
        retVal = max(results, key=lambda result: len(str(result)))

        # En uzun sonucu döndür
        return retVal

    def detect_horizontal_lines(self, pil_image, name="", threhold=1000):
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=threhold, minLineLength=10, maxLineGap=5)

        line_coords = []

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    if y1 == y2:  # Yatay çizgiler için
                        line_coords.append((x1, y1, x2, y2))
                        cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        line_coords = self.merge_close_lines(line_coords, 20)

        # Y koordinatına göre sıralama
        line_coords = sorted(line_coords, key=lambda x: x[1])

        marked_image = Image.fromarray(
            cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

        height = pil_image.height
        line_percents = [(line[1] / height) * 100 for line in line_coords]

        return {
            "name": name,
            "line_coordinates": line_coords,
            "line_percentages": line_percents,
            "marked_image": marked_image,
            "original_image": pil_image
        }

    @staticmethod
    def merge_close_lines(lines, max_distance=10):
        merged_lines = []

        while len(lines) > 0:
            current_line = lines[0]
            lines.remove(current_line)

            # Mevcut çizgiyi diğer çizgilerle karşılaştır
            close_lines = [current_line]
            for line in lines:
                if abs(line[1] - current_line[1]) <= max_distance:
                    close_lines.append(line)

            # Ortalama noktaları alarak yeni bir çizgi oluştur
            avg_x1 = sum([line[0] for line in close_lines]) / len(close_lines)
            avg_y1 = sum([line[1] for line in close_lines]) / len(close_lines)
            avg_x2 = sum([line[2] for line in close_lines]) / len(close_lines)
            avg_y2 = sum([line[3] for line in close_lines]) / len(close_lines)

            merged_lines.append(
                (int(avg_x1), int(avg_y1), int(avg_x2), int(avg_y2)))

            # Gruplanan çizgileri orijinal listeden çıkar
            for line in close_lines:
                if line in lines:
                    lines.remove(line)

        return merged_lines

    @staticmethod
    def erase_below_y(image, height_percent):
        # Resmin genişlik ve yüksekliğini al
        width, height = image.size

        # Yükseklik yüzdesini piksel cinsine çevirme
        y_coord = int(height * (height_percent / 100))

        # Resmi kırpma (crop) işlemi
        cropped_image = image.crop((0, 0, width, y_coord))

        return cropped_image
