if __name__ == '__main__':
    try:
        file = open('results.txt')
        line = file.readline()
        coef = float(line.strip().split(',')[0])
        slope = float(line.strip().split(',')[1])
        
        user_input = float(input('What value would you like to predict: '))
        result = coef + (user_input * slope)
        print('The predicted value is: ', round(result, 2))
    except:
        print('Some error ocurred. Make sure you trained the model first and the input is correct')