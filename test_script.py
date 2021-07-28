from datetime import datetime
import time
start = datetime.now()
time.sleep(5)
end = datetime.now()
print('it takes', end-start)