import argparse
from classifier.predict import predict as classify
from classifier.server import serve
from classifier.train import train_model as train_classifier

parser = argparse.ArgumentParser(description='Command-line tools for NLP tasks.')
subparsers = parser.add_subparsers(dest="command", help='Sub-command help')

# train
# example: python main.py train classifier
train_parser = subparsers.add_parser('train', help='Train a model')
train_parser.add_argument('model', type=str, help='The model to train')
train_parser.add_argument('--cont', action='store_true', help='Continue training the model')

# classify
# example: python main.py classify --n 3 "John Doe"
classify_parser = subparsers.add_parser('classify', help='Classify a person\'s name')
classify_parser.add_argument('name', type=str, help='The name to classify')
classify_parser.add_argument('--n', type=int, default=3, help='Number of predictions to show')

# server
# example: python main.py server
server_parser = subparsers.add_parser('server', help='Serve the model')

args = parser.parse_args()

if args.command == 'train':

    print('Training the model %s' % args.model)

    if args.model == 'classifier':
        train_classifier(args.cont)

elif args.command == 'classify':
    classify(args.name, args.n)

elif args.command == 'server':
    serve()

else:
    print('Unknown command: %s' % args.command)
    parser.print_help()

