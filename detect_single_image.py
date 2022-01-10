from gender_and_age import GenderAndAge

img_path = '/Users/aneira/noticias/data/tv24horas_2021_11_09_18_frame_87580.png'

gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
gaa.detect_single_image(img_path)
