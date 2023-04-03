#!/usr/local/bin/python
from __future__ import print_function
import os
import collections
from random import uniform
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_math_ops import maximum
from tensorflow.python.keras import backend
from tensorflow.python.ops.init_ops import Initializer
from tensorflow.python.util import nest
from tensorflow.contrib.layers import xavier_initializer
import numpy as np




_MSINStateTuple = collections.namedtuple("MSINStateTuple", ("h", "v"))

class MSINStateTuple(_MSINStateTuple):
  """Tuple used by LSTM Cells for `state_size`, `zero_state`, and output state.
  Stores two elements: `(c, h)`, in that order.
  Only used when `state_is_tuple=True`.
  """
  __slots__ = ()

  @property
  def dtype(self):
    (h, v) = self
    if h.dtype == v.dtype:
        return h.dtype
    else:
        raise TypeError("Inconsistent internal state: %s vs %s" %
                      (str(h.dtype), str(v.dtype)))
    
class MSINCell(object):        

    def __init__(self, input_size, num_units, v_size, max_n_msgs, forget_bias=1.0):
        super(MSINCell, self).__init__()

        self._input_sizes = input_size
        self._v_size = v_size
        self._num_units = num_units
        self._forget_bias = forget_bias
        self.max_n_msgs = max_n_msgs

        #v
        '''
        self.W_sa = tf.Variable(tf.random_normal(shape=[v_size, num_units],stddev=0.01,mean=0,dtype=tf.float32))
        self.W_ha = tf.Variable(tf.random_normal(shape=[num_units, num_units],stddev=0.01,mean=0,dtype=tf.float32))
        '''
        self.W_sa = tf.get_variable('W_sa',shape=[v_size, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.W_ha = tf.get_variable('W_ha',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_a = tf.Variable(tf.zeros(num_units), dtype=tf.float32)
        self.v_a = tf.Variable(tf.zeros(num_units), dtype=tf.float32)


        #f
        self.W_f = tf.get_variable('W_f',shape=[v_size, num_units],dtype=tf.float32, initializer=xavier_initializer())
        self.W_hf = tf.get_variable('W_hf',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_f = tf.Variable(tf.zeros(num_units), dtype=tf.float32)
        
        #o
        self.W_o = tf.get_variable('W_o',shape=[v_size, num_units],dtype=tf.float32, initializer=xavier_initializer())
        self.W_ho = tf.get_variable('W_ho',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_o = tf.Variable(tf.zeros(num_units), dtype=tf.float32)
        #t
        self.W_t = tf.get_variable('W_t',shape=[v_size, num_units],dtype=tf.float32, initializer=xavier_initializer())
        self.W_ht = tf.get_variable('W_ht',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_t = tf.Variable(tf.zeros(num_units), dtype=tf.float32)


        #k
        
        self.W_k = tf.get_variable('W_k',shape=[input_size, num_units],dtype=tf.float32, initializer=xavier_initializer())
        self.W_hk = tf.get_variable('W_hk',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.W_vk = tf.get_variable('W_vk',shape=[v_size, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_k = tf.Variable(tf.zeros(num_units), dtype=tf.float32)
        

        #hx
        self.W_s = tf.get_variable('W_s',shape=[input_size, num_units],dtype=tf.float32, initializer=xavier_initializer())
        self.W_hs = tf.get_variable('W_hs',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_s = tf.Variable(tf.zeros(num_units), dtype=tf.float32)
        
        #hv
        self.W_v = tf.get_variable('W_v',shape=[v_size, num_units],dtype=tf.float32, initializer=xavier_initializer())
        self.W_hv = tf.get_variable('W_hv',shape=[num_units, num_units],dtype=tf.float32,initializer=xavier_initializer())
        self.b_v = tf.Variable(tf.zeros(num_units), dtype=tf.float32)



    def _zero_state_tensors(self, state_size, batch_size, dtype):
 

        def get_state_shape(s):
    
            c = MSIN()._concat(batch_size, s)
            size = array_ops.zeros(c, dtype=dtype)

            return size

        return nest.map_structure(get_state_shape, state_size)
    
    def zero_state(self, batch_size, dtype):
        state_size = self.state_size
    
        with backend.name_scope(type(self).__name__ + "ZeroState"):
            output = self._zero_state_tensors(state_size, batch_size, dtype)
        return output

    @property
    def state_size(self):
        return MSINStateTuple(self._num_units, self._v_size)

    @property
    def output_size(self):
        return self._num_units

    @property
    def P_size(self):
        return self.max_n_msgs

    def call(self, X, S, state):
        # Parameters of gates are concatenated into one multiply for efficiency.
        h, v = state
    
        a = tf.tanh(tf.matmul(S, tf.reshape(tf.tile(self.W_sa, [tf.shape(S)[0],1]), [tf.shape(S)[0], self.W_sa.shape[0], self.W_sa.shape[1]])) + tf.reshape(tf.tile((tf.matmul(h, self.W_ha) + self.b_a), [tf.shape(S)[1], 1]), [tf.shape(S)[0], tf.shape(S)[1], self.W_sa.shape[1]]))
        P = tf.nn.softmax(tf.transpose(tf.matmul(a, tf.reshape(tf.tile(self.v_a, [tf.shape(S)[0]]), [tf.shape(S)[0] , self._num_units, 1])), perm = [0, 2, 1]))
        
        new_P = tf.squeeze(P, 1)

        text = tf.squeeze(tf.matmul(P, S), 1)
        F = tf.sigmoid(tf.matmul(text, self.W_f) + tf.matmul(h, self.W_hf)  + self.b_f)
        O = tf.sigmoid(tf.matmul(text, self.W_o) + tf.matmul(h, self.W_ho)  + self.b_o)
        text_new = tf.tanh(tf.matmul(text, self.W_t) + tf.matmul(h, self.W_ht)  + self.b_t)
        V = F * v + O * text_new

        k = tf.sigmoid(tf.matmul(X, self.W_k) + tf.matmul(h, self.W_hk) + tf.matmul(V, self.W_vk) + self.b_k)
        
        hx = tf.tanh(tf.matmul(X, self.W_s) + tf.matmul(h, self.W_hs)  + self.b_s)
        hv = tf.tanh(tf.matmul(h, self.W_hv) + tf.matmul(V, self.W_v) + self.b_v)

        H = (1 - k) * hv + k * hx

        new_state = MSINStateTuple(H, V)
        

        return H, new_P, new_state




class MSIN(object):
    
    def _concat(self,prefix, suffix, static=False):
        if isinstance(prefix, ops.Tensor):
            p = prefix
            p_static = tensor_util.constant_value(prefix)
            if p.shape.ndims == 0:
                p = array_ops.expand_dims(p, 0)
            elif p.shape.ndims != 1:
                raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
        else:
            p = tensor_shape.TensorShape(prefix)
            p_static = p.as_list() if p.ndims is not None else None
            p = (
                constant_op.constant(p.as_list(), dtype=dtypes.int32)
                if p.is_fully_defined() else None)
        if isinstance(suffix, ops.Tensor):
            s = suffix
            s_static = tensor_util.constant_value(suffix)
            if s.shape.ndims == 0:
                s = array_ops.expand_dims(s, 0)
            elif s.shape.ndims != 1:
                raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
        else:
            s = tensor_shape.TensorShape(suffix)
            s_static = s.as_list() if s.ndims is not None else None
            s = (
                constant_op.constant(s.as_list(), dtype=dtypes.int32)
                if s.is_fully_defined() else None)

        if static:
            shape = tensor_shape.TensorShape(p_static).concatenate(s_static)
            shape = shape.as_list() if shape.ndims is not None else None
        else:
            if p is None or s is None:
                raise ValueError("Provided a prefix or suffix of None: %s and %s" %
                       (prefix, suffix))
            shape = array_ops.concat((p, s), 0)
        return shape


    def _infer_state_dtype(self, explicit_dtype, state):
  
        if explicit_dtype is not None:
            return explicit_dtype
        elif nest.is_sequence(state):
            inferred_dtypes = [element.dtype for element in nest.flatten(state)]
            if not inferred_dtypes:
                raise ValueError("Unable to infer dtype from empty state.")
            all_same = all(x == inferred_dtypes[0] for x in inferred_dtypes)
            if not all_same:
                raise ValueError(
                    "State has tensors of different inferred_dtypes. Unable to infer a "
                    "single representative dtype.")
            return inferred_dtypes[0]
        else:
            return state.dtype

    def _maybe_tensor_shape_from_tensor(self,shape):
        if isinstance(shape, ops.Tensor):
            return tensor_shape.as_shape(tensor_util.constant_value(shape))
        else:
            return 

    def _msin_step(self,time,
              sequence_length,
              min_sequence_length,
              max_sequence_length,
              zero_output,
              zero_P,
              state,
              call_cell,
              state_size,
              skip_conditionals=False):


        # Convert state to a list for ease of use
        flat_state = nest.flatten(state)
        flat_zero_output = nest.flatten(zero_output)
        flat_zero_P = nest.flatten(zero_P)
        
        # Vector describing which batch entries are finished.
        copy_cond = time >= sequence_length

        def _copy_one_through(output, new_output):
            # TensorArray and scalar get passed through.
            if isinstance(output, tensor_array_ops.TensorArray):
                return new_output
            if len(output.shape.as_list()) == 0:
                return new_output
            # Otherwise propagate the old or the new value.
            with ops.colocate_with(new_output):
                return array_ops.where(copy_cond, output, new_output)

        def _copy_some_through(flat_new_output, flat_new_P, flat_new_state):
            # Use broadcasting select to determine which values should get
            # the previous state & zero output, and which values should get
            # a calculated state & output.
            flat_new_output = [
                _copy_one_through(zero_output, new_output)
                for zero_output, new_output in zip(flat_zero_output, flat_new_output)
            ]

            
            flat_new_P = [
                _copy_one_through(zero_P, new_P)
                for zero_P, new_P in zip(flat_zero_P, flat_new_P)
            ]

            flat_new_state = [
                _copy_one_through(state, new_state)
                for state, new_state in zip(flat_state, flat_new_state)
            ]
            return flat_new_output + flat_new_P + flat_new_state

        def _maybe_copy_some_through():
            """Run msin step.  Pass through either no or some past state."""
            new_output, new_P,  new_state = call_cell()

            nest.assert_same_structure(zero_output, new_output)
            nest.assert_same_structure(zero_P, new_P)
            nest.assert_same_structure(state, new_state)

            flat_new_state = nest.flatten(new_state)
            flat_new_output = nest.flatten(new_output)
            flat_new_P = nest.flatten(new_P)
            
            
            return control_flow_ops.cond(
                # if t < min_seq_len: calculate and return everything
                time < min_sequence_length,
                lambda: flat_new_output + flat_new_P + flat_new_state,
                # else copy some of it through
                lambda: _copy_some_through(flat_new_output, flat_new_P, flat_new_state))

        # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
        # but benefits from removing cond() and its gradient.  We should
        # profile with and without this switch here.
        if skip_conditionals:
            # Instead of using conditionals, perform the selective copy at all time
            # steps.  This is faster when max_seq_len is equal to the number of unrolls
            # (which is typical for dynamic_msin).
            new_output, new_P, new_state = call_cell()
            nest.assert_same_structure(zero_output, new_output)
            nest.assert_same_structure(zero_P, new_P)
            nest.assert_same_structure(state, new_state)
            new_state = nest.flatten(new_state)
            new_output = nest.flatten(new_output)
            new_P = nest.flatten(new_P)
            final_output_and_state = _copy_some_through(new_output, new_P, new_state)
        else:
            empty_update = lambda: flat_zero_output + flat_zero_P +flat_state
            final_output_and_state = control_flow_ops.cond(
                # if t >= max_seq_len: copy all state through, output zeros
                time >= max_sequence_length,
                empty_update,
                # otherwise calculation is required: copy some or all of it through
                _maybe_copy_some_through)

        if len(final_output_and_state) != len(flat_zero_output) + len(flat_zero_P) + len(flat_state):
            raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")

        final_output = final_output_and_state[:len(flat_zero_output)]
        final_P = final_output_and_state[len(flat_zero_output):len(flat_zero_output)+len(flat_zero_P)]
        final_state = final_output_and_state[len(flat_zero_output)+len(flat_zero_P):]

        for output, flat_output in zip(final_output, flat_zero_output):
            output.set_shape(flat_output.get_shape())
        for P, flat_P in zip(final_P, flat_zero_P):
            P.set_shape(flat_P.get_shape())
        for substate, flat_substate in zip(final_state, flat_state):
            if not isinstance(substate, tensor_array_ops.TensorArray):
                substate.set_shape(flat_substate.get_shape())

        final_output = nest.pack_sequence_as(
            structure=zero_output, flat_sequence=final_output)
        final_P = nest.pack_sequence_as(
            structure=zero_P, flat_sequence=final_P)
        final_state = nest.pack_sequence_as(
            structure=state, flat_sequence=final_state)

        return final_output, final_P, final_state



    def _dynamic_msin_loop(self,
                      cell,
                      inputs,
                      s_inputs,
                      initial_state,
                      parallel_iterations,
                      swap_memory,
                      sequence_length=None,
                      dtype=None):
 
        state = initial_state
        assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

        state_size = cell.state_size

        flat_input = nest.flatten(inputs)
        flat_s_inputs = nest.flatten(s_inputs)
        flat_output_size = nest.flatten(cell.output_size)
        flat_P_size = nest.flatten(cell.P_size)

        # Construct an initial output
        input_shape = array_ops.shape(flat_input[0])
        time_steps = input_shape[0]
        batch_size = self._best_effort_input_batch_size(flat_input)

        self.inputs_got_shape = tuple(
        input_.get_shape().with_rank_at_least(3) for input_ in flat_input)
        self.s_inputs_got_shape = tuple(
        input_.get_shape().with_rank_at_least(4) for input_ in flat_s_inputs)
        


        const_time_steps, const_batch_size = self.inputs_got_shape[0].as_list()[:2]

        for shape in self.inputs_got_shape:
            if not shape[2:].is_fully_defined():
                raise ValueError(
                    "Input size (depth of inputs) must be accessible via shape inference,"
                    " but saw value None.")
            got_time_steps = shape.dims[0].value
            got_batch_size = shape.dims[1].value
            if const_time_steps != got_time_steps:
                raise ValueError(
                    "Time steps is not the same for all the elements in the input in a "
                    "batch.")
            if const_batch_size != got_batch_size:
                raise ValueError(
                    "Batch_size is not the same for all the elements in the input.")
        # Prepare dynamic conditional copying of state & output

        def _create_zero_arrays(size):
            size = self._concat(batch_size, size)
            return array_ops.zeros(
                array_ops.stack(size), self._infer_state_dtype(dtype, state))


        flat_zero_output = tuple(
            _create_zero_arrays(output) for output in flat_output_size)
        
        zero_output = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=flat_zero_output)
  

        flat_zero_P = tuple(
            _create_zero_arrays(P) for P in flat_P_size)
 
        zero_P = nest.pack_sequence_as(
        structure=cell.P_size, flat_sequence=flat_zero_P)
        
        if sequence_length is not None:
            min_sequence_length = math_ops.reduce_min(sequence_length)
            max_sequence_length = math_ops.reduce_max(sequence_length)
        else:
            max_sequence_length = time_steps

        time = array_ops.constant(0, dtype=dtypes.int32, name="time")

        with ops.name_scope("dynamic_msin") as scope:
            base_name = scope

        def _create_ta(name, element_shape, dtype):
            return tensor_array_ops.TensorArray(
                dtype=dtype,
                size=time_steps,
                element_shape=element_shape,
                tensor_array_name=base_name + name)

        self.in_graph_mode= True
        if self.in_graph_mode:
            output_ta = tuple(
                _create_ta(
                    "output_%d" % i,
                    element_shape=(
                        tensor_shape.TensorShape([const_batch_size]).concatenate(
                            self._maybe_tensor_shape_from_tensor(out_size))),
                    dtype=self._infer_state_dtype(dtype, state))
                for i, out_size in enumerate(flat_output_size))
            P_ta = tuple(
                _create_ta(
                    "P_%d" % i,
                    element_shape=(
                        tensor_shape.TensorShape([const_batch_size]).concatenate(
                            self._maybe_tensor_shape_from_tensor(P_out_size))),
                    dtype=self._infer_state_dtype(dtype, state))
                for i, P_out_size in enumerate(flat_P_size))
            input_ta = tuple(
                _create_ta(
                    "input_%d" % i,
                    element_shape=flat_input_i.shape[1:],
                    dtype=flat_input_i.dtype)
                for i, flat_input_i in enumerate(flat_input))
            self.input_ta = tuple(
                ta.unstack(input_) for ta, input_ in zip(input_ta, flat_input))
            s_input_ta = tuple(
                _create_ta(
                    "input_%d" % i,
                    element_shape=flat_s_input_i.shape[1:],
                    dtype=flat_s_input_i.dtype)
                for i, flat_s_input_i in enumerate(flat_s_inputs))
            self.s_input_ta = tuple(
                ta.unstack(s_input_) for ta, s_input_ in zip(s_input_ta, flat_s_inputs))
        else:
            output_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_output_size)))
            P_ta = tuple([0 for _ in range(time_steps.numpy())]
                      for i in range(len(flat_P_size)))
            self.input_ta = flat_input
            self.s_input_ta = flat_s_inputs
            

        def _time_step(time, output_ta_t, P_ta_t, state):
    

            if self.in_graph_mode:
                input_t = tuple(ta.read(time) for ta in self.input_ta)
                # Restore some shape information
                for input_, shape in zip(input_t, self.inputs_got_shape):
                    input_.set_shape(shape[1:])

                s_input_t = tuple(ta.read(time) for ta in self.s_input_ta)
                for s_input_, shape in zip(s_input_t, self.s_inputs_got_shape):
                    s_input_.set_shape(shape[1:])

            else:
                input_t = tuple(ta[time.numpy()] for ta in self.input_ta)
                s_input_t = tuple(ta[time.numpy()] for ta in self.s_input_ta)


            input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
            s_input_t = nest.pack_sequence_as(structure=s_inputs, flat_sequence=s_input_t)
            
            
            # Keras rnn cells only accept state as list, even if it's a single tensor.
            
            call_cell = lambda: cell.call(input_t, s_input_t, state)

            if sequence_length is not None:
                (output, P, new_state) = self._msin_step(
                    time=time,
                    sequence_length=sequence_length,
                    min_sequence_length=min_sequence_length,
                    max_sequence_length=max_sequence_length,
                    zero_output=zero_output,
                    zero_P=zero_P,
                    state=state,
                    call_cell=call_cell,
                    state_size=state_size,
                    skip_conditionals=True)
            else:
                (output, P, new_state) = call_cell()

            # Pack state if using state tuples
            output = nest.flatten(output)
            P = nest.flatten(P)


            if self.in_graph_mode:
                output_ta_t = tuple(
                ta.write(time, out) for ta, out in zip(output_ta_t, output))
                P_ta_t = tuple(
                da.write(time, P_out) for da, P_out in zip(P_ta_t, P))
            else:
                for ta, out in zip(output_ta_t, output):
                    ta[time.numpy()] = out
                for da, P_out in zip(P_ta_t, P):
                    da[time.numpy()] = P_out

            return (time + 1, output_ta_t, P_ta_t, new_state)



        _, output_final_ta, P_final_ta, final_state = control_flow_ops.while_loop(
            cond=lambda time, *_: time < time_steps,
            body=_time_step,
            loop_vars=(time, output_ta, P_ta, state),
            parallel_iterations=parallel_iterations,
            #maximum_iterations=time_steps,
            swap_memory=swap_memory)

        
        # Unpack final output if not using output tuples.
        if self.in_graph_mode:
            final_outputs = tuple(ta.stack() for ta in output_final_ta)
            # Restore some shape information
            for output, output_size in zip(final_outputs, flat_output_size):
                shape = self._concat([const_time_steps, const_batch_size],
                      output_size,
                      static=True)
                output.set_shape(shape)


            final_P = tuple(da.stack() for da in P_final_ta)
            # Restore some shape information
            for P, P_size in zip(final_P, flat_P_size):
                P_shape = self._concat([const_time_steps, const_batch_size],
                      P_size,
                      static=True)
                P.set_shape(P_shape)
        else:
            final_outputs = output_final_ta
            final_P = P_final_ta

        final_outputs = nest.pack_sequence_as(
            structure=cell.output_size, flat_sequence=final_outputs)
        final_P = nest.pack_sequence_as(
            structure=cell.P_size, flat_sequence=final_P)
        if not self.in_graph_mode:
            final_outputs = nest.map_structure_up_to(
                cell.output_size, lambda x: array_ops.stack(x, axis=0), final_outputs)
            final_P = nest.map_structure_up_to(
                cell.P_size, lambda x: array_ops.stack(x, axis=0), final_P)


        

        return (final_outputs, final_P, final_state)

        
    def _transpose_batch_time(self,x):
        """Transposes the batch and time dimensions of a Tensor.
            If the input tensor has rank < 2 it returns the original tensor. Retains as
            much of the static shape information as possible.
            Args:
                x: A Tensor.
            Returns:
                x transposed along the first two dimensions.
        """
        x_static_shape = x.get_shape()
        if len(x_static_shape.as_list()) is not None and len(x_static_shape.as_list())<2:
            return x

        x_rank = array_ops.rank(x)
        x_t = array_ops.transpose(
            x, array_ops.concat(([1, 0], math_ops.range(2, x_rank)), axis=0))
        x_t.set_shape(
            tensor_shape.TensorShape(
            [x_static_shape.dims[1].value,
            x_static_shape.dims[0].value]).concatenate(x_static_shape[2:]))
        return x_t

    def _best_effort_input_batch_size(self,flat_input):

        for input_ in flat_input:
            shape = input_.shape
            if len(shape.as_list()) is None:
                continue
            if len(shape.as_list()) < 2:
                raise ValueError("Expected input tensor %s to have rank at least 2" %
                       input_)
            batch_size = shape.dims[1].value
            if batch_size is not None:
                return batch_size
        # Fallback to the dynamic batch size of the first input.
        return array_ops.shape(flat_input[0])[1]


    def dynamic_msin(self,
                cell,
                inputs,
                s_inputs,
                sequence_length=None,
                initial_state=None,
                dtype=None,
                parallel_iterations=None,
                swap_memory=False,
                time_major=False,
                scope=None):


            


            # By default, time_major==False and inputs are batch-major: shaped
            #   [batch, time, depth]
            # For internal calculations, we transpose to [time, batch, depth]
        flat_input = nest.flatten(inputs)
        flat_s_inputs = nest.flatten(s_inputs)

        if not time_major:
            # (B,T,D) => (T,B,D)
            flat_input = [ops.convert_to_tensor(input_) for input_ in flat_input]
            flat_input = tuple(self._transpose_batch_time(input_) for input_ in flat_input)
            flat_s_inputs = [ops.convert_to_tensor(s_input_) for s_input_ in flat_s_inputs]
            flat_s_inputs = tuple(self._transpose_batch_time(s_input_) for s_input_ in flat_s_inputs)

        parallel_iterations = parallel_iterations or 32
        if sequence_length is not None:
            sequence_length = math_ops.cast(sequence_length, dtypes.int32)
            if len(sequence_length.get_shape().as_list()) not in (None, 1):
                raise ValueError(
                    "sequence_length must be a vector of length batch_size, "
                    "but saw shape: %s" % sequence_length.get_shape())
            sequence_length = array_ops.identity(  # Just to find it in the graph.
                    sequence_length,
                    name="sequence_length")

        with vs.variable_scope(scope or "msin") as varscope:

            # Create a new scope in which the caching device is either
            # determined by the parent scope, or is set to place the cached
            # Variable using the same placement as for the rest of the RNN.
  
            if varscope.caching_device is None:
                varscope.set_caching_device(lambda op: op.device)
                
            batch_size = self._best_effort_input_batch_size(flat_input)


            if initial_state is not None:
                state = initial_state
            else:
                if not dtype:
                    raise ValueError("If there is no initial_state, you must give a dtype.")
                if getattr(cell, "get_initial_state", None) is not None:
                    state = cell.get_initial_state(
                        inputs=None, s_inputs=None, batch_size=batch_size, dtype=dtype)
                else:
                    state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = array_ops.shape(x)
            packed_shape = array_ops.stack(shape)
            return control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)), [
                    "Expected shape for Tensor %s is " % x.name, packed_shape,
                    " but saw shape: ", x_shape
                ])



        inputs = nest.pack_sequence_as(structure=inputs, flat_sequence=flat_input)
        s_inputs = nest.pack_sequence_as(structure=s_inputs, flat_sequence=flat_s_inputs)

        (outputs, P, final_state) = self._dynamic_msin_loop(
            cell,
            inputs,
            s_inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype)

        # Outputs of _dynamic_msin_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            outputs = nest.map_structure(self._transpose_batch_time, outputs)
            P = nest.map_structure(self._transpose_batch_time, P)

        return (outputs, P, final_state)


    
    
    