import sys
import logging
import logger


def error_messages_info(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = '\nScript filename: {0}\nLine#: {1}\nError: {2}'.format(file_name, exc_tb.tb_lineno, str(error))
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_messages_info(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message  