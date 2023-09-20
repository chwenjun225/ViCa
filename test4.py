<<<<<<< HEAD
import time
=======
import time 
>>>>>>> origin/master
import random
from barcode import EAN13
from barcode.writer import ImageWriter


def generate_barcode():
    count_index = 0

<<<<<<< HEAD
    while count_index < 1000:
        number_in_barcode = str(random.random()).replace(".", "")
        my_code = EAN13(number_in_barcode, writer=ImageWriter())
        my_code.save(f"Data/SR_Barcode_Images/{count_index}")
        count_index += 1


if __name__ == "__main__":

    RUN_ = 1
    if 1 == RUN_:
        generate_barcode()
=======
    while count_index < 1000: 
        number_in_barcode = str(random.random()).replace(".", "")
        my_code = EAN13(number_in_barcode, writer=ImageWriter())
        my_code.save(f"OpenSR_barcode_images/{count_index}")
        count_index += 1


if __name__=="__main__":

    RUN_ = 1
    if 1 == RUN_:

        start_time = time.time()

        generate_barcode()
        
        print("The time of execution of above program is :", 
        (end-start) * 10**3, "ms")
>>>>>>> origin/master
