from preprocessing import get_preprocessing


def get_prediction(data, model):

    """[[close_went_up, close_went_down]]"""
    data = get_preprocessing(data)
    answer = model.predict(data)[0]
    print(answer)
    if answer > 0.5:
        print('up')
        return 'up'
    else:
        print('down')
        return 'down'


# print(get_prediction([[random.randint(100, 150) for i in range(3)] for ii in range(14)]))
'''WARNING:tensorflow:6 out of the last 6 calls to <function TensorFlowTrainer.make_predict_function.<locals>.one_step_on_data_distributed at 0x00000295317319E0> triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to (1) creating @tf.function repeatedly in a loop, (2) passing tensors with different shapes, (3) passing Python objects instead of tensors. For (1), please define your @tf.function outside of the loop. For (2), @tf.function has reduce_retracing=True option that can avoid unnecessary retracing. For (3), please refer to https://www.tensorflow.org/guide/function#controlling_retracing and https://www.tensorflow.org/api_docs/python/tf/function for  more details.'''
