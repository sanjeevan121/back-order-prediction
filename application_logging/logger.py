from datetime import datetime

class App_Logger:
    def __init__(self):
        pass

    async def log(self, file_object, log_message):
        self.now = datetime.now()
        self.date = self.now.date()
        self.current_time = self.now.strftime("%H:%M:%S")
       # logger = Logger.with_default_handlers(name='my-logger')
        await file_object.logger.write(
            str(self.date) + "/" + str(self.current_time) + "\t\t" + log_message + "\n")
        await file_object.flush()