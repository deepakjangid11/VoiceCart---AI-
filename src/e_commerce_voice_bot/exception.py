import sys


class CustomException(Exception):

    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.line_number = exc_tb.tb_lineno
        else:
            self.file_name = "Unknown"
            self.line_number = 0

        self.error_message = str(error_message)

    def __str__(self):
        return (
            f"Error in {self.file_name}, "
            f"line {self.line_number}: "
            f"{self.error_message}"
        )