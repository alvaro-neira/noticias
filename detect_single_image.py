from gender_and_age import GenderAndAge

img_path = '/Users/aneira/noticias/data/tv24horas_2021_10_30_18_frame_128163.png'

gaa = GenderAndAge('/Users/aneira/noticias/Gender-and-Age-Detection')
gaa.detect_single_image(img_path)
