import json
import os
from json import JSONDecodeError

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


class GSheets:
    ascii_pos = 64

    @staticmethod
    def __number2excel(n, base):
        return "" + chr(n + GSheets.ascii_pos + 26 - base);

    @staticmethod
    def change_base_excel(n):
        base = 26
        rem = n % base
        ret_val = "" + GSheets.__number2excel(rem, base);

        while n > 0:
            n = n // base
            if n == 0:
                break
            n = n - 1
            rem = n % base
            ret_val = GSheets.__number2excel(rem, base) + ret_val
        return ret_val
