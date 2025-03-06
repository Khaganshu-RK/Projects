import sys

class ProjectException(Exception):
    def __init__(self, error_message, error_detail:sys):
        self.error_message = error_message
        _,_,exc_tb = error_detail.exc_info()

        self.line_number = exc_tb.tb_lineno
        self.filename = exc_tb.tb_frame.f_code.co_filename

        self.error_message = f"{self.error_message} \nError occured in {self.filename} at line number {self.line_number}"


    def __str__(self):
        return self.error_message