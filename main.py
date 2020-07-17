from Concept_Module import *


data = pickle.load(open("datafile", "rb"))
ans = pickle.load(open("ansfile", "rb"))
torch.manual_seed(0)

modality_dims = {"audio":[36, 20, 10], "flow":[8,6, 4], "haptics" :[13,10, 5],"vibro":[70,50, 30],
          "surf":[90,60, 40], "fingers":[6,4, 3], "color":[134,100, 50], "weight": [1, 1, 1], "height": [1,1,1]}

model_file = "conceptqa_trained"
epochs = 3
conceptqa = ConceptQA(modality_dims, 768, 200, 100)
crit = nn.CrossEntropyLoss()
lr = 5.0
#optim = torch.optim.SGD(conceptqa.parameters(), lr=lr)
optim = torch.optim.Adam(conceptqa.parameters(), 1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optim, 1, gamma=0.95)
prev = np.float("inf")

for e in range(epochs):
    start = time.time()
    train(conceptqa, data[:int(len(data)/5)], optim, crit)
    val_loss, preds = evaluate(conceptqa, data[:int(len(data)/5)], crit)
    if val_loss < prev:
        prev = val_loss
        best_model = conceptqa
    scheduler.step()
    elapsed = time.time() - start
    print("Epoch:", start, val_loss)
    start = time.time()


torch.save(best_model.state_dict(), model_file)