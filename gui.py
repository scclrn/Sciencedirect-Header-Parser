import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import fitz  # PyMuPDF
from scripts.donut_model import DonutModel  # DonutModel'ı utils.py dosyasından import edin
import pandas as pd
import ast
from pdf2image import convert_from_path, convert_from_bytes

class PDFManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Manager")
        self.geometry("1024x768")

        # Sol çerçeve (PDF listesi için)
        self.left_frame = tk.Frame(self, width=200, bg='lightgrey')
        self.left_frame.pack(side='left', fill='y')

        # Orta çerçeve (PDF önizleme için)
        self.middle_frame = tk.Frame(self, bg='white')
        self.middle_frame.pack(side='left', fill='both', expand=True)

        # Sağ çerçeve (Genel ayarlar için)
        # Genişliği 150 piksel yapın
        self.right_frame = tk.Frame(self, bg='lightgrey', width=150)
        self.right_frame.pack(side='right', fill='y')

        # PDF listesi ve kaydırma çubuğu
        self.pdf_listbox = tk.Listbox(self.left_frame)
        self.pdf_listbox.pack(side='top', fill='both',
                              expand=True, padx=10, pady=10)
        self.scrollbar = tk.Scrollbar(
            self.left_frame, orient='vertical', command=self.pdf_listbox.yview)
        self.scrollbar.pack(side='right', fill='y')
        self.pdf_listbox.config(yscrollcommand=self.scrollbar.set)

        # PDF seçildiğinde önizleme göster
        self.pdf_listbox.bind("<<ListboxSelect>>", self.preview_pdf)

        # Düğmeler
        self.button_frame = tk.Frame(self.left_frame, bg='lightgrey')
        self.button_frame.pack(side='bottom', fill='x', pady=5)

        self.add_button = tk.Button(
            self.button_frame, text="PDF Ekle", command=self.add_pdf)
        self.add_button.pack(side='left', padx=5)

        self.remove_button = tk.Button(
            self.button_frame, text="PDF Çıkar", command=self.remove_pdf)
        self.remove_button.pack(side='right', padx=5)

        # PDF listesi
        self.pdf_files = []

        # PDF önizleme Canvas'ı
        self.preview_canvas = tk.Canvas(self.middle_frame, bg='white')
        self.preview_canvas.pack(fill='both', expand=True)

        # Kaydırma çubukları
        self.h_scrollbar = tk.Scrollbar(
            self.preview_canvas, orient='horizontal', command=self.preview_canvas.xview)
        self.h_scrollbar.pack(side='bottom', fill='x')
        self.v_scrollbar = tk.Scrollbar(
            self.preview_canvas, orient='vertical', command=self.preview_canvas.yview)
        self.v_scrollbar.pack(side='right', fill='y')

        self.preview_canvas.config(
            xscrollcommand=self.h_scrollbar.set, yscrollcommand=self.v_scrollbar.set)

        # Yüklenmiş PDF önizleme görüntüsü
        self.preview_image = None

        # Genel Ayarlar Düğmeleri (Dikey düzenleme)
        self.process_button = tk.Button(
            self.right_frame, text="PDF'leri İşle", command=self.process_pdfs)
        self.process_button.pack(side='top', fill='x', padx=10, pady=5)

        self.save_csv_button = tk.Button(
            self.right_frame, text="CSV Olarak Kaydet", command=self.save_as_csv)
        self.save_csv_button.pack(side='top', fill='x', padx=10, pady=5)

        self.save_excel_button = tk.Button(
            self.right_frame, text="Excel Olarak Kaydet", command=self.save_as_excel)
        self.save_excel_button.pack(side='top', fill='x', padx=10, pady=5)

        # İşlem Sonucu Gösterim Alanı (Text Area)
        self.result_text_area = tk.Text(
            self.right_frame, wrap='word', height=10)
        self.result_text_area.pack(
            side='top', fill='both', padx=10, pady=5, expand=True)

        # DonutModel sınıfını oluştur
        self.donut_model = DonutModel()

    def add_pdf(self):
        files = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if files:
            for file in files:
                if file not in self.pdf_files:
                    self.pdf_files.append(file)
                    self.pdf_listbox.insert(tk.END, file.split('/')[-1])
            self.preview_pdf(None)

    def remove_pdf(self):
        selected_index = self.pdf_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            del self.pdf_files[index]
            self.pdf_listbox.delete(index)
            self.preview_canvas.delete("all")
        else:
            messagebox.showwarning("Uyarı", "Lütfen çıkarılacak bir PDF seçin.")

    def preview_pdf(self, event=None):
        selected_index = self.pdf_listbox.curselection()
        if selected_index:
            index = selected_index[0]
            pdf_path = self.pdf_files[index]
            self.show_pdf_preview(pdf_path)

    def show_pdf_preview(self, pdf_path):
        # PDF'i aç
        pdf_document = fitz.open(pdf_path)
        page = pdf_document.load_page(0)  # İlk sayfayı yükle

        # Sayfayı görüntü olarak dışa aktar
        pix = page.get_pixmap()
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Görüntüyü yeniden boyutlandır, oranı koruyarak
        canvas_width, canvas_height = self.preview_canvas.winfo_width(), self.preview_canvas.winfo_height()
        ratio = min(canvas_width / pix.width, canvas_height / pix.height)
        new_size = (int(pix.width * ratio), int(pix.height * ratio))
        resized_image = image.resize(new_size, Image.LANCZOS)

        # Görüntüyü Canvas üzerinde göster
        self.preview_image = ImageTk.PhotoImage(resized_image)
        self.preview_canvas.create_image(0, 0, anchor='nw', image=self.preview_image)

        # Canvas ve kaydırma alanını güncelle
        self.preview_canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))

    def process_pdfs(self):
        # Loading ekranını göster
        loading_screen = tk.Toplevel(self)
        loading_screen.title("PDF'ler İşleniyor")
        loading_screen.geometry("300x100")
        loading_label = tk.Label(loading_screen, text="PDF'ler işleniyor, lütfen bekleyin...", padx=20, pady=20)
        loading_label.pack()

        self.update()  # GUI'nin güncellenmesi için
        self.result_text_area.delete(1.0, tk.END)  # Önceki metni temizleyin

        for pdf_file in self.pdf_files:
            pdf_text = self.process_pdf_with_model(pdf_file)
            if pdf_text:
                self.result_text_area.insert(tk.END, f"PDF Dosyası: {pdf_file}\n{pdf_text}\n\n")

        loading_screen.destroy()  # İşlem tamamlandığında loading ekranını kapat

        messagebox.showinfo("Bilgi", "PDF'ler başarıyla işlendi.")

    def process_pdf_with_model(self, pdf_path):
        try:
            # PDF dosyasını aç
            images = convert_from_path(pdf_path)
            image = images[0]

            # DonutModel ile kırpılmış görüntüyü işleme
            processed_text = self.donut_model.preprocess_pdf_image(
                image)
            return processed_text

        except Exception as e:
            messagebox.showerror("Hata", f"PDF işleme sırasında bir hata oluştu: {str(e)}")
            return ""

    def convert_to_df(self):

        # result_text_area içeriğini satır satır okuyup işleyelim
        content = self.result_text_area.get(1.0, tk.END)

        data_list = [elem for elem in content.split('\n') if elem != '']

        retVal = []
        for i in range(0, len(data_list), 2):
            # data_list[i] dosya yolunu, data_list[i+1] JSON verisini içerir
            file_path = data_list[i].replace('PDF Dosyası: ', '')
            
            # JSON verisini güvenli şekilde çözümle
            try:
                jsondata = ast.literal_eval(data_list[i+1])
            except (SyntaxError, ValueError) as e:
                print(f"Hata: JSON verisini çözümlemede sorun oluştu: {str(e)}")
                continue
            
            # 'header' anahtarı varsa sadece o kısmı al, yoksa boş bir sözlük oluştur
            if isinstance(jsondata, dict) and 'header' in jsondata:
                jsondata = jsondata['header']
            elif isinstance(jsondata, dict) and len(jsondata.keys()) > 0:
                jsondata = jsondata
            else:
                jsondata = {}

            if isinstance(jsondata, dict) == False:
                jsondata = {}
            
            # Veriye erişirken anahtarların varlığını kontrol et
            data = {
                'Dosya Yolu': file_path,
                'Yayınlanma Tarihi': str(jsondata.get('publication_date', '')),
                'Yayınlandığı Dergi İsmi': str(jsondata.get('journal_name', '')),
                'Makale İsmi': str(jsondata.get('article_name', '')),
                'Yazar İsmi': str(jsondata.get('author_name', ''))
            }
            retVal.append(data)
        df = pd.DataFrame(retVal)
        return df

    def save_as_csv(self):
        if not self.pdf_files:
            messagebox.showwarning("Uyarı", "Lütfen önce PDF dosyası ekleyin.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            df = self.convert_to_df()
            df.to_csv(file_path, index=False)
            messagebox.showinfo("Bilgi", "CSV dosyası başarıyla kaydedildi.")
        else:
            messagebox.showwarning("Uyarı", "Lütfen bir dosya yolu belirterek tekrar deneyin.")

    def save_as_excel(self):
        if not self.pdf_files:
            messagebox.showwarning("Uyarı", "Lütfen önce PDF dosyası ekleyin.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("XLSX files", "*.xlsx")])
        if file_path:
            df = self.convert_to_df()
            df.to_excel(file_path, index=False)
            messagebox.showinfo("Bilgi", "Excel dosyası başarıyla kaydedildi.")
        else:
            messagebox.showwarning("Uyarı", "Lütfen bir dosya yolu belirterek tekrar deneyin.")


if __name__ == "__main__":
    app = PDFManagerApp()
    app.mainloop()
