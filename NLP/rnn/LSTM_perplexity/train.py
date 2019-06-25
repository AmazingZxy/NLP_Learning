# _*_ encoding=utf8 _*_

import numpy as np
import tensorflow as tf
import os
from get_batch import read_data,make_batches
from model import PTBModel
from model_config import FLAGS

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_epoch(session, model, batches, train_op, output_log, step):
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)
    for x,y in batches:
        cost,state,_ = session.run(
            [model.cost,model.final_state,train_op],
            feed_dict =  {model.input_data:x, model.targets:y,
                          model.initial_state:state}
        )
        total_costs += cost
        iters += model.num_steps

        if output_log and step % 100 == 0:
            print("After %d steps,perplexity is %.3f" %(step, np.exp(total_costs/iters)))

        step += 1

    return step,np.exp(total_costs / iters)


def main():
    initialzer = tf.random_uniform_initializer(-0.05,0.05)

    with tf.variable_scope("language_model", reuse=None,initializer=initialzer):
        train_model = PTBModel(True, FLAGS.TRAIN_BATCH_SIZE, FLAGS.TRAIN_NUM_STEP)

    with tf.variable_scope("language_model", reuse=True,initializer=initialzer):
        eval_model = PTBModel(False, FLAGS.EVAL_BATCH_SIZE, FLAGS.EVAL_NUM_STEP)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        train_batches = make_batches(
            read_data(FLAGS.TRAIN_DATA), FLAGS.TRAIN_BATCH_SIZE, FLAGS.TRAIN_NUM_STEP
        )

        eval_batches = make_batches(
            read_data(FLAGS.EVAL_DATA), FLAGS.EVAL_BATCH_SIZE, FLAGS.EVAL_NUM_STEP
        )

        test_batches = make_batches(
            read_data(FLAGS.TEST_DATA), FLAGS.EVAL_BATCH_SIZE, FLAGS.EVAL_NUM_STEP
        )

        step = 0
        for i in range(FLAGS.NUM_EPOCH):
            print("in iteration :%d " % (i+1))
            step,train_pplx  = run_epoch(sess, train_model, train_batches,
                                         train_model.train_op,True, step)
            print("Epoch: %d Train perplexity:%.3f" % (i+1,train_pplx))

            step, eval_pplx = run_epoch(sess, eval_model, eval_batches,
                                         tf.no_op(), False, 0)
            print("Epoch: %d Eval perplexity:%.3f" % (i + 1, eval_pplx))

        step, test_pplx = run_epoch(sess, eval_model, test_batches,
                                    tf.no_op(), False, 0)
        print("Test perplexity:%.3f" % test_pplx)

if __name__ == '__main__':
    main()
