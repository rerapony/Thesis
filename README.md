# Thesis
medium-term forecasting of Earth's outer radiation belt's relativistic electrons fluxes based on observations of solar coronal holes

script*.py - внутри программы волняется обработка входных данных и реализуется обучение ансамбля нейронных сетей, предсказания которых усредняются и записываются в отдельный файл
также возвращает показатели метрик MSE и R2, validation loss

принимает на вход директории output и checkpoint - для хранения возвращаемых результатов и лучшей модели за всё время обучения
директория parameters - физические параметры и их временное погружение, которые подаются на вход
mode - применяет два значения None и SW - без использования солнечного ветра и с использованием

data_coronal_holes.csv - необработанная табличка с необработанными данными 

further updates are on the way...
