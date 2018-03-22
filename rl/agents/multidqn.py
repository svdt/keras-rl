from __future__ import division
import warnings, re

import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input, Layer, Dense
from keras.utils.vis_utils import plot_model

from rl.core import Agent
from rl.policy import EpsGreedyMQPolicy, GreedyMQPolicy
from rl.util import *
from PIL import Image

def mean_q(y_true, y_pred):
    return K.mean(K.max(y_pred, axis=-1))



def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str
    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)


class AbstractDQNAgent(Agent):
    """Write me
    """
    def __init__(self, nb_actions, memory, gamma=.99, batch_size=32, nb_steps_warmup=1000,
                 train_interval=1, memory_interval=1, target_model_update=20,
                 delta_range=None, delta_clip=np.inf, custom_model_objects={}, **kwargs):
        super(AbstractDQNAgent, self).__init__(**kwargs)

        # Soft vs hard target model updates.
        if target_model_update < 0:
            raise ValueError('`target_model_update` must be >= 0.')
        elif target_model_update >= 1:
            # Hard update every `target_model_update` steps.
            target_model_update = int(target_model_update)
        else:
            # Soft update with `(1 - target_model_update) * old + target_model_update * new`.
            target_model_update = float(target_model_update)

        if delta_range is not None:
            warnings.warn('`delta_range` is deprecated. Please use `delta_clip` instead, which takes a single scalar. For now we\'re falling back to `delta_range[1] = {}`'.format(delta_range[1]))
            delta_clip = delta_range[1]

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.custom_model_objects = custom_model_objects

        # Related objects.
        self.memory = memory

        # State.
        self.compiled = False

    def process_state_batch(self, batch):
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch, self.input_shape)

    def process_reward_batch(self, reward):
        if self.processor is None:
            return batch
        return self.processor.process_reward_batch(reward)

    def compute_batch_q_values(self, agent, state_batch):
        batch = self.process_state_batch(state_batch)
        # only show first input of first observation
        if agent.showconv and agent.step % 2 == 0:
            self.print_img(self.mynet,4,batch)
        q_values = self.model.predict_on_batch(batch)
        q_values = [q_values] if type(q_values) is not list else q_values
        #assert q_values.shape == (len(state_batch), self.nb_actions)
        return q_values

    def compute_q_values(self, agent, state):
        q_values = self.compute_batch_q_values(agent, [state])
        assert tuple([len(q_shape[0]) for q_shape in q_values]) == self.nb_actions
        return q_values

    def get_config(self):
        return {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
        }

# An implementation of the DQN agent as described in Mnih (2013) and Mnih (2015).
# http://arxiv.org/pdf/1312.5602.pdf
# http://arxiv.org/abs/1509.06461
class MDQNAgent(AbstractDQNAgent):
    """Write me
    """
    def __init__(self, model, training=True, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False,
                 dueling_type='avg', *args, **kwargs):
        super(MDQNAgent, self).__init__(*args, **kwargs)

        # Validate (important) input.
        #if hasattr(model.output, '__len__') and len(model.output) > 1:
        #    raise ValueError('Model "{}" has more than one output. DQN expects a model that has a single output.'.format(model))
        #if model.output._keras_shape != (None, self.nb_actions):
        #    raise ValueError('Model output "{}" has invalid shape. DQN expects a model that has one dimension for each action, in this case {}.'.format(model.output, self.nb_actions))

        # Parameters.
        self.enable_double_dqn = enable_double_dqn
        self.enable_dueling_network = enable_dueling_network
        self.dueling_type = dueling_type
        if self.enable_dueling_network:
            # get the second last layer of the model, abandon the last layer
            layer = model.layers[-2]
            nb_action = model.output._keras_shape[-1]
            # layer y has a shape (nb_action+1,)
            # y[:,0] represents V(s;theta)
            # y[:,1:] represents A(s,a;theta)
            y = Dense(nb_action + 1, activation='linear')(layer.output)
            # caculate the Q(s,a;theta)
            # dueling_type == 'avg'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-Avg_a(A(s,a;theta)))
            # dueling_type == 'max'
            # Q(s,a;theta) = V(s;theta) + (A(s,a;theta)-max_a(A(s,a;theta)))
            # dueling_type == 'naive'
            # Q(s,a;theta) = V(s;theta) + A(s,a;theta)
            if self.dueling_type == 'avg':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'max':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], keepdims=True), output_shape=(nb_action,))(y)
            elif self.dueling_type == 'naive':
                outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(nb_action,))(y)
            else:
                assert False, "dueling_type must be one of {'avg','max','naive'}"

            model = Model(inputs=model.input, outputs=outputlayer)

        # Related objects.
        self.model = model
        if policy is None:
            policy = EpsGreedyMQPolicy()
        if test_policy is None:
            test_policy = GreedyMQPolicy()
        self.policy = policy
        self.test_policy = test_policy
        self.policy._set_agent(self)
        self.test_policy._set_agent(self)
        self.training = training
        # State.
        self.reset_states()

    def print_img(self,model,l,img):
        self.mynet.set_weights(self.model.get_weights())
        inps = [self.mynet.input] if type(self.mynet.input) is not list else self.mynet.input
        for idx,inp in enumerate(inps):
            layer = self.mynet.layers[l+idx]
            model = Model(inputs=inp, outputs=layer.output)
            out = model.predict(np.array([img[idx][0]]))
            for i,fil in enumerate(np.transpose(out[0])):
                Image.fromarray(255*fil).convert('L').save("conv-inp"+str(idx)+"/filter"+str(i)+".png")

    def get_config(self):
        config = super(DQNAgent, self).get_config()
        config['enable_double_dqn'] = self.enable_double_dqn
        config['dueling_type'] = self.dueling_type
        config['enable_dueling_network'] = self.enable_dueling_network
        config['model'] = get_object_config(self.model)
        config['policy'] = get_object_config(self.policy)
        config['test_policy'] = get_object_config(self.test_policy)
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    def compile(self, optimizer, metrics=[]):
        metrics += [mean_q]  # register default metrics
        self.mynet = clone_model(self.model, self.custom_model_objects)
        self.mynet.compile(optimizer='sgd', loss='mse')

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        def clipped_masked_error(args):
            y_true, y_pred, mask = args
            loss = huber_loss(y_true, y_pred, self.delta_clip)
            loss *= mask  # apply element-wise mask
            return K.sum(loss, axis=-1)


        def avg_loss(args):
            loss = K.sum([l for l in args])/len(args)
            return loss

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_preds = [self.model.output] if type(self.model.output) is not list else self.model.output
        y_trues = [Input(name='y_true'+str(i), shape=(n_action,)) for i,n_action in enumerate(self.nb_actions)]
        masks = [Input(name='mask'+str(i), shape=(n_action,)) for i,n_action in enumerate(self.nb_actions)]
        losses_out = [Lambda(clipped_masked_error, output_shape=(1,), name='loss-'+str(i))([y_preds[i], y_trues[i], masks[i]]) for i in range(len(y_preds))]
        #loss_out = Lambda(avg_loss, output_shape=(1,), name='total-loss')(losses_out)

        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        trainable_model = Model(inputs=ins + y_trues + masks, outputs=losses_out+y_preds)

        #assert len(trainable_model.output_names) == 2
        combined_metrics = {name: metrics for name in self.model.output_names}
        losses = [
            *[lambda y_true, y_pred: y_pred for i in losses_out],  # loss is computed in Lambda layer
            *[lambda y_true, y_pred: K.zeros_like(y_pred) for i in y_preds]  # we only include this for the metrics
        ]
        plot_model(trainable_model, to_file='trainable_model_plot.png', show_shapes=True, show_layer_names=True)
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model
        self.model._make_predict_function()
        self.trainable_model._make_predict_function()
        self.target_model._make_predict_function()

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        self.target_model.set_weights(self.model.get_weights())

    def forward(self, agent, observation):
        # Select an action.
        state = agent.memory.get_recent_state(observation)
        q_values = self.compute_q_values(agent,state)
        if self.training:
            action = agent.policy.select_action(q_values=q_values)
        else:
            action = agent.test_policy.select_action(q_values=q_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    def backward_thread(self, reward, terminal, queue):
        queue.put(self.backward(reward, terminal))

    def backward(self, agent, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            agent.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)
        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = agent.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = self.process_reward_batch(np.array(reward_batch))
            action_batch = self.process_reward_batch(np.array(action_batch))
            assert reward_batch.shape == (len(self.model.outputs),self.batch_size)
            assert terminal1_batch.shape == (self.batch_size,)
            assert action_batch.shape == reward_batch.shape

            # Compute Q values for mini-batch update.
            if self.enable_double_dqn:
                # According to the paper "Deep Reinforcement Learning with Double Q-learning"
                # (van Hasselt et al., 2015), in Double DQN, the online network predicts the actions
                # while the target network is used to estimate the Q value.
                q_values = self.model.predict_on_batch(state1_batch)
                q_values = [q_values] if type(q_values) is not list else q_values
                actions = []
                for i,q_v in enumerate(q_values):
                    assert q_v.shape == (self.batch_size, self.nb_actions[i])
                    action = np.argmax(q_v, axis=1)
                    actions.append(action)
                    assert action.shape == (self.batch_size,)

                # Now, estimate Q values using the target network but select the values with the
                # highest Q value wrt to the online model (as computed above).
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                target_q_values = [target_q_values] if type(target_q_values) is not list else target_q_values
                q_batch = []
                for i,t_q_v in enumerate(target_q_values):
                    assert t_q_v.shape == (self.batch_size, self.nb_actions[i])
                    q_batch.append(t_q_v[range(self.batch_size), actions[i]])
            else:
                # Compute the q_values given state1, and extract the maximum for each sample in the batch.
                # We perform this prediction on the target_model instead of the model for reasons
                # outlined in Mnih (2015). In short: it makes the algorithm more stable.
                target_q_values = self.target_model.predict_on_batch(state1_batch)
                target_q_values = [target_q_values] if type(target_q_values) is not list else target_q_values
                q_batch = []
                for i,t_q_v in enumerate(target_q_values):
                    assert t_q_v.shape == (self.batch_size, self.nb_actions[i])
                    q_batch.append(np.max(t_q_v, axis=1).flatten())
            assert len(q_batch) == len(self.model.outputs)

            targets = [np.zeros((self.batch_size, self.nb_actions[i])) for i in range(len(self.model.outputs))]
            dummy_targets = [np.zeros((self.batch_size,)) for i in range(len(self.model.outputs))]
            masks = [np.zeros((self.batch_size, self.nb_actions[i])) for i in range(len(self.model.outputs))]

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = np.array([self.gamma * q_b for q_b in q_batch])
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch
            assert discounted_reward_batch.shape == reward_batch.shape
            Rs = reward_batch + discounted_reward_batch
            for _idx, (_targets, _masks, _Rs, _action_batch) in enumerate(zip(targets, masks, Rs, action_batch)):
                for idx, (target, mask, R, action) in enumerate(zip(_targets, _masks, _Rs, _action_batch)):
                    target[action] = R  # update action with estimated accumulated reward
                    dummy_targets[_idx][idx] = R
                    mask[action] = 1.  # enable loss for this specific action
                    target.astype('float32')
                    mask.astype('float32')


            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            #state0_batch = [state0_batch] if type(self.model.input) is not list else state0_batch
            metrics = self.trainable_model.train_on_batch(state0_batch + targets + masks, dummy_targets + targets)
            #metrics = [metric for idx, metric in enumerate(metrics) if idx not in (1, 2)]  # throw away individual losses
            #print(self.policy.metrics_names)
            metrics += agent.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.

        #assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.model.output_names
        model_metrics = [name for name in self.trainable_model.metrics_names if name not in dummy_output_name]
        model_metrics = [multireplace(name,{don: 'action-'+str(i) for i,don in enumerate(dummy_output_name)}) for idx, name in enumerate(model_metrics)]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)
