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
            for sample in state['train_loader']:
                self.hook('on_start_train_iteration', state)
                state['input'] = sample  # input : torch_geometric.data.Data
                state['label'] = None  # on_sampleで設定する
                self.hook('on_sample', state)

                def closure():
                    output = state['network'](state['input'])
                    loss = criterion(output, state['label'])
                    loss.backward()
                    state['output'] = output
                    state['loss'] = loss
                    self.hook('on_forward', state)
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
            for sample in state['val_loader']:
                self.hook('on_start_val_iteration', state)
                state['input'] = sample
                state['label'] = sample.y
                self.hook('on_sample', state)

                def closure():
                    output = state['network'](state['input'])
                    loss = criterion(output, state['label'])
                    state['output'] = output
                    state['loss'] = loss
                    self.hook('on_forward', state)
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
