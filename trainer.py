import torch
import torch.nn.functional as F



def train(model, device, train_loader, optimizer, criterion, epoch, pruner):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if pruner:
            pruner.prune(model=model, stage=0, update_masks=False, verbose=False)

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.sampler),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.sampler)
        accuracy = 100. * correct / len(test_loader.sampler)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
            test_loss, correct, len(test_loader.sampler), accuracy))

    return test_loss, accuracy


def adjust_learning_rate(optimizer, learning_rate):

    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate


def run_training(args, model, device, train_loader, test_loader, optimizer, criterion, fname, pruner=None, overfit=False, val_loader=None):

    t_loss, t_acc = None, None

    step_lr = int(args.num_epochs/args.lr_decay_step)

    for epoch in range(args.num_epochs):

        if args.decay_lr == 1:
            if epoch % step_lr == 0 and epoch > 0:
                args.lr /= 10
                adjust_learning_rate(optimizer, args.lr)
                print("\nAdjusted learning rate: {} ...\n".format(args.lr))

        train(model, device, train_loader, optimizer, criterion, epoch, pruner=pruner)
        loss, acc = test(model, device, test_loader)

        if args.save == 1:
            torch.save(model.state_dict(), 'models/' + args.data + '/' + fname + '.pth')
            print("saved model...")

        if overfit:
            print('\nGetting Training Accuracy...')
            t_loss, t_acc = test(model, device, train_loader)

            if t_acc == 100.0 or abs(t_acc - 99.99) < 0.001:
                break

        if val_loader:
            v_loss, v_acc = test(model, device, val_loader)
            if overfit:
                print("\nTrain Accuracy:\t\t" + str(t_acc))
            print("Validation Accuracy:\t" + str(v_acc))
            print("Test Accuracy:\t\t" + str(acc) + "\n")
