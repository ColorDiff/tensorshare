Examples
========

:class:`tensorshare <tensorshare.ts.TensorShare>`
-------------------------------------------------

1. Start the tensorshare server on a free port of your choice.

.. code-block:: python

   from tensorshare import run_server

   run_server(8000)


2. Connect your clients

.. code-block:: python

   from tensorshare import TensorShare
   from threading import Thread
   import torch


   def do_something(ts):
       ts.put('data_bin', dict(a=torch.randn(2, 2), version=0))  # Not blocking
       ts.peek('data_bin', lambda x: print('peek data_bin', x))  # Get data in the bin and it

       # Alternatively specify a callback
       def cb_get_data(response):
           ts.get('xyz', lambda x: print('get xyz', x))  # This is called as soon as the server responds
       ts.put('xyz', dict(a=torch.randn(2, 2), version=0), callback=cb_get_data)

       # Listening to data bins
       def cb_bin_updated(msg):
           if 'bin' in msg.keys():
               print('Bin', msg['bin'], 'updated. Num items: ', msg['n_items'])
       ts.listen('xyz', cb_bin_updated)    # Listen to changes in bin xyz, calling cb_bin_updated whenever it happens
       ts.append('xyz', torch.randn(2, 2))    # Appending will not overwrite data in the bin, but append to it, creating a list
       ts.append('xyz', torch.randn(2, 2))
       ts.get('xyz', lambda x: print('get xyz', x))   # Get removes data in the bin, which will also emit a notification.
       ts.put('xyz', torch.randn(2, 2))

       def cb_print_and_exit(resp):
           print(resp)
           ts.stop()
       ts.list(cb_print_and_exit)  # List the number of items in all bins


   tmp = TensorShare('localhost', 8000)
   Thread(target=do_something, args=(tmp, ), daemon=True).start()
   tmp.start() # This is blocking and must be started in the main thread.

Since `tensorshare` relies on `twisted` for communcations, the main-thread will always be occupied by `twisted`'s reactor.
Hence, all code using `tensorshare` must be run in separate threads.

For DDRL, you may want to use the `RLTrainer` and `RLWorker` classes.
These support exchange of network parameters and aggregation of self-play or rollout data and filtering by parameter version (to avoid off-policy samples).

.. _rl_trainer_example:

:class:`RLTrainer <tensorshare.ts.RLTrainer>`
---------------------------------------------

.. code-block:: python

   from tensorshare import RLTrainer
   import torch
   from threading import Thread
   from time import sleep


   def train(ts):
       net = torch.nn.Linear(10, 2)
       try:
           for i_iter in range(10):
               ts.publish_parameters(net.state_dict())

               # Aggregate a batch
               batch_size = 16
               data_buffer = torch.zeros(batch_size, 10)
               i = 0
               while i < batch_size:
                   data = ts.get_data()
                   for item in data:
                       data_buffer[i, :] = item
                       i += 1
                       if i == batch_size:
                           break
                   sleep(.5)

               # Train on the batch
               loss = net(data_buffer).mean()
               print('Iteration', i_iter, 'Loss:', loss)
               loss.backward()
               with torch.no_grad():
                   net.weight -= 0.001 * net.weight.grad
                   net.weight.grad *= 0

               # Publish new parameters
               ts.publish_parameters(net.state_dict())
       finally:
           ts.stop()

   # Also host the server. In a real application it's better to have a dedicated process host the server.
   tmp = RLTrainer('localhost', 8000, host_server=True, filter_version=False)
   Thread(target=train, args=(tmp, ), daemon=True).start()
   tmp.start()

In this example, the Trainer does not filter data by parameter version, which means that some samples will be off-policy.
If you want to filter by parameter version, set `filter_version=True`.

:class:`RLWorker <tensorshare.ts.RLWorker>`
-------------------------------------------

.. code-block:: python

   from tensorshare import RLWorker
   import torch
   from threading import Thread
   from time import sleep

   def generate_rollouts(ts):
       net = torch.nn.Linear(10, 2)

       try:
           # Load latest published parameters
           params = ts.get_parameters()
           if params is None:  # In case no parameters were available on the server
               params = ts.await_new_parameters(timeout=None)   # We wait until they are available
           net.load_state_dict(params)

           for _ in range(300):
               # Generate a rollout
               x = torch.randn(1, 10)
               ts.add_data(x)
               sleep(.2)
               # Load newest parameters
               params = ts.get_parameters()    # This returns the latest published parameters
               net.load_state_dict(params)
       finally:
           ts.stop()


   tmp = RLWorker('localhost', 8000)
   Thread(target=generate_rollouts, args=(tmp, ), daemon=True).start()
   tmp.start()

