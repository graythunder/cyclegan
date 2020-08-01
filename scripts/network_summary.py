from torchsummary import summary
from model import *

summary(Generator(), (3,256,256))
summary(Discriminator(), (3,256,256))

