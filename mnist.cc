#include <caffe2/core/init.h>
#include "modules/ModelHelpers/blob.h"
//#include "caffe2/util/net.h"
#include "modules/ModelHelpers/net.h"
#include "modules/ModelHelpers/model.h"
#include "caffe2/core/logging.h"
// #include "modules/cvplot/cvplot.h"

#ifdef WITH_CUDA
#include <caffe2/core/context_gpu.h>
#endif

C10_DEFINE_string(train_db, "res/mnist-train-nchw-leveldb","The given path to the training leveldb.");
C10_DEFINE_string(test_db, "res/mnist-test-nchw-leveldb","The given path to the testing leveldb.");
C10_DEFINE_int(iters, 100, "The of training runs.");
C10_DEFINE_int(test_runs, 50, "The of test runs.");
C10_DEFINE_bool(force_cpu, false, "Only use CPU, no CUDA.");
C10_DEFINE_bool(display, false, "Display graphical training info.");

namespace caffe2 {

// >> def AddInput(model, batch_size, db, db_type):
void AddInput(ModelUtil &model, int batch_size, const std::string &db,
              const std::string &db_type) {
  // Setup database connection
  model.init.AddCreateDbOp("dbreader", db_type, db);
  model.predict.AddInput("dbreader");

  // >>> data_uint8, label = model.TensorProtosDBInput([], ["data_uint8",
  // "label"], batch_size=batch_size, db=db, db_type=db_type)
  model.predict.AddTensorProtosDbInputOp("dbreader", "data_uint8", "label",
                                         batch_size);

  // >>> data = model.Cast(data_uint8, "data", to=core.DataType.FLOAT)
  model.predict.AddCastOp("data_uint8", "data", TensorProto_DataType_FLOAT);

  // >>> data = model.Scale(data, data, scale=float(1./256))
  model.predict.AddScaleOp("data", "data", 1.f / 256);

  // >>> data = model.StopGradient(data, data)
  model.predict.AddStopGradientOp("data");
}

// def AddLeNetModel(model, data):
void AddLeNetModel(ModelUtil &model, bool test) {
  // >>> conv1 = brew.conv(model, data, 'conv1', dim_in=1, dim_out=20, kernel=5)
  model.AddConvOps("data", "conv1", 1, 20, 1, 0, 5, test);

  // >>> pool1 = brew.max_pool(model, conv1, 'pool1', kernel=2, stride=2)
  model.predict.AddMaxPoolOp("conv1", "pool1", 2, 0, 2);

  // >>> conv2 = brew.conv(model, pool1, 'conv2', dim_in=20, dim_out=50,
  // kernel=5)
  model.AddConvOps("pool1", "conv2", 20, 50, 1, 0, 5, test);

  // >>> pool2 = brew.max_pool(model, conv2, 'pool2', kernel=2, stride=2)
  model.predict.AddMaxPoolOp("conv2", "pool2", 2, 0, 2);

  // >>> fc3 = brew.fc(model, pool2, 'fc3', dim_in=50 * 4 * 4, dim_out=500)
  model.AddFcOps("pool2", "fc3", 800, 500, test);

  // >>> fc3 = brew.relu(model, fc3, fc3)
  model.predict.AddReluOp("fc3", "fc3");

  // >>> pred = brew.fc(model, fc3, 'pred', 500, 10)
  model.AddFcOps("fc3", "pred", 500, 10, test);

  // >>> softmax = brew.softmax(model, pred, 'softmax')
  model.predict.AddSoftmaxOp("pred", "softmax");
}

// def AddAccuracy(model, softmax, label):
void AddAccuracy(ModelUtil &model) {
  // >>> accuracy = model.Accuracy([softmax, label], "accuracy")
  model.predict.AddAccuracyOp("softmax", "label", "accuracy");

  if (FLAGS_display) {
    model.predict.AddTimePlotOp("accuracy");
  }

  // >>> ITER = model.Iter("iter")
  model.AddIterOps();
}

// // >>> def AddTrainingOperators(model, softmax, label):
// void AddTrainingOperators(ModelUtil &model) {
//   // >>> xent = model.LabelCrossEntropy([softmax, label], 'xent')
//   model.predict.AddLabelCrossEntropyOp("softmax", "label", "xent");

//   // >>> loss = model.AveragedLoss(xent, "loss")
//   model.predict.AddAveragedLossOp("xent", "loss");

//   if (FLAGS_display) {
//     model.predict.AddShowWorstOp("softmax", "label", "data", 256, 0);
//     model.predict.AddTimePlotOp("loss");
//   }

//   // >>> AddAccuracy(model, softmax, label)
//   AddAccuracy(model);

//   // >>> model.AddGradientOperators([loss])
//   model.predict.AddConstantFillWithOp(1.0, "loss", "loss_grad");
//   model.predict.AddGradientOps();

//   // >>> LR = model.LearningRate(ITER, "LR", base_lr=-0.1, policy="step",
//   // stepsize=1, gamma=0.999 )
//   model.predict.AddLearningRateOp("iter", "LR", 0.1);

//   // >>> ONE = model.param_init_net.ConstantFill([], "ONE", shape=[1],
//   // value=1.0)
//   model.init.AddConstantFillOp({1}, 1.f, "ONE");
//   model.predict.AddInput("ONE");

//   // >>> for param in model.params:
//   for (auto param : model.Params()) {
//     // >>> param_grad = model.param_to_grad[param]
//     // >>> model.WeightedSum([param, ONE, param_grad, LR], param)
//     model.predict.AddWeightedSumOp({param, "ONE", param + "_grad", "LR"},
//                                    param);
//   }

//   // Checkpoint causes problems on subsequent runs
//   // >>> model.Checkpoint([ITER] + model.params, [],
//   // std::vector<std::string> inputs({"iter"});
//   // inputs.insert(inputs.end(), params.begin(), params.end());
//   // model.predict.AddCheckpointOp(inputs, 20, "leveldb",
//   //                         "mnist_lenet_checkpoint_%05d.leveldb");
// }

// // >>> def AddBookkeepingOperators(model):
// void AddBookkeepingOperators(ModelUtil &model) {
//   // >>> model.Print('accuracy', [], to_file=1)
//   model.predict.AddPrintOp("accuracy", true);

//   // >>> model.Print('loss', [], to_file=1)
//   model.predict.AddPrintOp("loss", true);

//   // >>> for param in model.params:
//   for (auto param : model.Params()) {
//     // >>> model.Summarize(param, [], to_file=1)
//     model.predict.AddSummarizeOp(param, true);

//     // >>> model.Summarize(model.param_to_grad[param], [], to_file=1)
//     model.predict.AddSummarizeOp(param + "_grad", true);
//   }
// }

// void run() {
//   std::cout << std::endl;
//   std::cout << "## Caffe2 MNIST Tutorial ##" << std::endl;
//   std::cout << "https://caffe2.ai/docs/tutorial-MNIST.html" << std::endl;
//   std::cout << std::endl;

//   if (!std::ifstream(FLAGS_train_db).good() ||
//       !std::ifstream(FLAGS_test_db).good()) {
//     std::cerr << "error: MNIST database missing: "
//               << (std::ifstream(FLAGS_train_db).good() ? FLAGS_test_db
//                                                        : FLAGS_train_db)
//               << std::endl;
//     std::cerr << "Make sure to first run ./script/download_resource.sh"
//               << std::endl;
//     return;
//   }

//   std::cout << "train-db: " << FLAGS_train_db << std::endl;
//   std::cout << "test-db: " << FLAGS_test_db << std::endl;
//   std::cout << "iters: " << FLAGS_iters << std::endl;
//   std::cout << "test-runs: " << FLAGS_test_runs << std::endl;
//   std::cout << "force-cpu: " << (FLAGS_force_cpu ? "true" : "false")
//             << std::endl;
//   std::cout << "display: " << (FLAGS_display ? "true" : "false") << std::endl;

// #ifdef WITH_CUDA
//   if (!FLAGS_force_cpu) {
//     DeviceOption option;
//     option.set_device_type(CUDA);
//     new CUDAContext(option);
//     std::cout << std::endl << "using CUDA" << std::endl;
//   }
// #endif


  //可以使用
  // if (FLAGS_display)
  //  {
  //   cvplot::Window::current("Caffe2 MNIST Tutorial");
  //   cvplot::moveWindow("undercertain", 0, 0);
  //   cvplot::resizeWindow("undercertain", 300, 300);
  //   cvplot::moveWindow("overcertain", 0, 300);
  //   cvplot::resizeWindow("overcertain", 300, 300);
  //   cvplot::moveWindow("accuracy", 300, 0);
  //   cvplot::resizeWindow("accuracy", 300, 300);
  //   cvplot::moveWindow("loss", 300, 300);
  //   cvplot::resizeWindow("loss", 300, 300);
  // }

//   // >>> from caffe2.python import core, cnn, net_drawer, workspace, visualize,
//   // brew
//   // >>> workspace.ResetWorkspace(root_folder)
//   Workspace workspace("tmp");

//   // >>> train_model = model_helper.ModelHelper(name="mnist_train",
//   // arg_scope={"order": "NCHW"})
//   NetDef train_init_model, train_predict_model;
//   ModelUtil train(train_init_model, train_predict_model, "mnist_train");

// //   // >>> data, label = AddInput(train_model, batch_size=64,
// //   // db=os.path.join(data_folder, 'mnist-train-nchw-leveldb'),
// //   // db_type='leveldb')
// //   AddInput(train, 64, FLAGS_train_db, "leveldb");

// //   // >>> softmax = AddLeNetModel(train_model, data)
// //   AddLeNetModel(train, false);

// //   // >>> AddTrainingOperators(train_model, softmax, label)
// //   AddTrainingOperators(train);

// //   // >>> AddBookkeepingOperators(train_model)
// //   AddBookkeepingOperators(train);

// //   // >>> test_model = model_helper.ModelHelper(name="mnist_test",
// //   // arg_scope=arg_scope, init_params=False)
// //   NetDef test_init_model, test_predict_model;
// //   ModelUtil test(test_init_model, test_predict_model, "mnist_test");

// //   // >>> data, label = AddInput(test_model, batch_size=100,
// //   // db=os.path.join(data_folder, 'mnist-test-nchw-leveldb'), db_type='leveldb')
// //   AddInput(test, 100, FLAGS_test_db, "leveldb");

// //   // >>> softmax = AddLeNetModel(test_model, data)
// //   AddLeNetModel(test, true);

// //   // >>> AddAccuracy(test_model, softmax, label)
// //   AddAccuracy(test);

// //   // >>> deploy_model = model_helper.ModelHelper(name="mnist_deploy",
// //   // arg_scope=arg_scope, init_params=False)
// //   NetDef deploy_init_model, deploy_predict_model;
// //   ModelUtil deploy(deploy_init_model, deploy_predict_model, "mnist_model");
// //   deploy.predict.AddInput("data");
// //   deploy.predict.AddOutput("softmax");

// //   // >>> AddLeNetModel(deploy_model, "data")
// //   AddLeNetModel(deploy, true);

// // #ifdef WITH_CUDA
// //   if (!FLAGS_force_cpu) {
// //     train.SetDeviceCUDA();
// //     test.SetDeviceCUDA();
// //   }
// // #endif

// //   std::cout << std::endl;

// //   // >>> workspace.RunNetOnce(train_model.param_init_net)
// //   CAFFE_ENFORCE(workspace.RunNetOnce(train.init.net));

// //   // >>> workspace.CreateNet(train_model.net)
// //   CAFFE_ENFORCE(workspace.CreateNet(train.predict.net));

// //   std::cout << "training.." << std::endl;

// //   // >>> for i in range(total_iters):
// //   for (auto i = 1; i <= FLAGS_iters; i++) {
// //     // >>> workspace.RunNet(train_model.net.Proto().name)
// //     CAFFE_ENFORCE(workspace.RunNet(train.predict.net.name()));

// //     // >>> accuracy[i] = workspace.FetchBlob('accuracy')
// //     // >>> loss[i] = workspace.FetchBlob('loss')
// //     // if (i % 10 == 0) {
// //     //   auto accuracy =
// //     //       BlobUtil(*workspace.GetBlob("accuracy")).Get().data<float>()[0];
// //     //   auto loss = BlobUtil(*workspace.GetBlob("loss")).Get().data<float>()[0];
// //     //   std::cout << "step: " << i << " loss: " << loss
// //     //             << " accuracy: " << accuracy << std::endl;
// //     // }
// //   }

//   std::cout << std::endl;
// }



}  // namespace caffe2

int main(int argc, char **argv) {
  caffe2::GlobalInit(&argc, &argv);
  // caffe2::run();
  google::protobuf::ShutdownProtobufLibrary();
  return 0;
}
