from gender_and_age import GenderAndAge

# img_path = '/Users/aneira/noticias/data/tv24horas_2022_01_06_10_frame_124425.png'
# img_path = '/Users/aneira/noticias/data/tv24horas_2022_01_06_12_frame_27000.png'
img_path = '/Users/aneira/noticias/data/tv24horas_2022_01_06_10_frame_10.png'

gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
gaa.detect_single_image(img_path)
