columns = ['Date', 'Season', 'Vụ nuôi', 'module_name', 'ao', 
           'Ngày thả', 'Time','Nhiệt độ', 'pH', 'Độ mặn', 
           'TDS', 'Độ đục', 'DO', 'Độ màu', 'Độ trong','Độ kiềm', 
           'Độ cứng','Loại ao', 'Công nghệ nuôi', 'area', 
           'Giống tôm', 'Tuổi tôm', 'Mực nước', 'Amoni', 
           'Nitrat', 'Nitrit', 'Silica', 'Canxi', 'Kali', 'Magie']

# input_col = ['Season', 'Ngày thả', 'Nhiệt độ', 'pH', 'Độ mặn', 
#            'TDS', 'Độ đục', 'DO', 'Độ màu', 'Độ trong', 
#            'Loại ao', 'Công nghệ nuôi', 'area', 
#            'Giống tôm', 'Tuổi tôm', 'Mực nước']


input_col = [
    'Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm', 
    'Ngày thả', 'Nhiệt độ', 'pH', 'Độ mặn', 
    #    'TDS', 'Độ đục', 'DO',
    'Độ trong', 
    'area', 
    'Tuổi tôm', 'Mực nước']

output_folder = "output"



categorical_col = ['Date','Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm','units']

categorical_usecol = [
    'Season', 'Loại ao', 'Công nghệ nuôi', 'Giống tôm'
    ]

# output_column = ['TAN', 'Nitrat', 'Nitrit', 'Silica', 'Canxi', 'Kali', 'Magie', 'Độ kiềm', 'Độ cứng']
output_column = ['Độ kiềm']
zscore_lim =  3
