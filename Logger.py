import os
import datetime
import codecs
import logging as log
class Logger:
    app_path = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(app_path, "Logs")
    location = os.path.join(
            log_path,
            "log-" + datetime.datetime.now().strftime("%Y-%m-%d") + ".log",
        )
    @staticmethod
    def create_log_folder():
        if not os.path.exists(Logger.log_path):
            os.makedirs(Logger.log_path)

    @staticmethod
    def write_log(message, log_type="Debug"):
        message = str(message).replace("\n", " ").replace("\r", " ").replace("\t", " ")
        # print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')} : {log_type} - {message}\n")
        Logger.create_log_folder()
        
        try:
            with codecs.open(Logger.location, "a", "utf-8") as writer:
                writer.write(
                    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S:%f')} : {log_type} - {message}\n"
                )
                writer.flush()
        except Exception as e:
            pass
            # print(f"Error writing to log file: {e}")

if __name__ == "__main__":
    Logger.write_log("Debug message.", "Debug")
    Logger.write_log("Error message.", "Exception")
