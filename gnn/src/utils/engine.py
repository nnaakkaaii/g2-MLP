import torch


class Engine:
    """reference : torchnet.engine.Engine
    """
    def __init__(self):
        self.hooks = {}

    def hook(self, name, state):
        if name in self.hooks:
            self.hooks[name](state)

    def train(self, network, train_loader, val_loader, max_epoch, optimizer, scheduler, criterion):
        state = {
            'network': network,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'max_epoch': max_epoch,
            'optimizer': optimizer,
            'criterion': criterion,
            'scheduler': scheduler,
            'epoch': 0,
            'iteration': 0,
            'train': True,
            'input': None,
            'label': None,
            'output': None,
            'loss': None,
        }

        self.hook('on_start_training', state)
        while state['epoch'] < state['max_epoch']:
            self.hook('on_start_epoch', state)
            self.hook('on_start_train_epoch', state)
            state['network'].train()
            state['train'] = True
            for sample in state['train_loader']:
                self.hook('on_start_train_iteration', state)
                state['input'] = sample  # input : torch_geometric.data.Data
                state['label'] = None  # on_sampleで設定する
                self.hook('on_sample', state)

                def closure():
                    output = state['network'](state['input'])
                    state['output'] = output
                    self.hook('on_forward', state)
                    loss = criterion(state['output'], state['label'])
                    loss.backward()
                    state['loss'] = loss
                    self.hook('on_backward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None
                    return loss

                state['optimizer'].zero_grad()
                state['optimizer'].step(closure)
                state['scheduler'].step()
                self.hook('on_update', state)
                self.hook('on_end_train_iteration', state)
                state['iteration'] += 1
            self.hook('on_end_train_epoch', state)

            self.hook('on_start_val_epoch', state)
            state['network'].eval()
            state['train'] = False
            for sample in state['val_loader']:
                self.hook('on_start_val_iteration', state)
                state['input'] = sample
                state['label'] = None
                self.hook('on_sample', state)

                def closure():
                    output = state['network'](state['input'])
                    state['output'] = output
                    self.hook('on_forward', state)
                    loss = criterion(state['output'], state['label'])
                    state['loss'] = loss
                    self.hook('on_backward', state)
                    # to free memory in save_for_backward
                    state['output'] = None
                    state['loss'] = None

                with torch.no_grad():
                    closure()

                self.hook('on_end_val_iteration', state)
                state['iteration'] += 1
            self.hook('on_end_val_epoch', state)
            self.hook('on_end_epoch', state)
            state['epoch'] += 1
        self.hook('on_end_training', state)
        return state

    def inference(self, network, test_loader):
        state = {
            'network': network,
            'test_loader': test_loader,
            'iteration': 0,
            'train': False,
            'input': None,
            'label': None,
            'output': None,
            'loss': None,
        }

        self.hook('on_start_test', state)
        state['network'].eval()
        state['train'] = False
        for sample in state['test_loader']:
            self.hook('on_start_test_iteration', state)
            state['input'] = sample
            state['label'] = None
            self.hook('on_sample', state)

            def closure():
                output = state['network'](state['input'])
                state['output'] = output
                self.hook('on_forward', state)
                # to free memory in save_for_backward
                state['output'] = None
                state['loss'] = None

            with torch.no_grad():
                closure()

            self.hook('on_end_test_iteration', state)
            state['iteration'] += 1
        self.hook('on_end_test', state)
        return state