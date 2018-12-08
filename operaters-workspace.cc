#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/proto/caffe2.pb.h>
namespace caffe2 {
    void print(Blob* blob, const std::string& name) {
        Tensor* tensor = BlobGetMutableTensor(blob, caffe2::DeviceType::CPU);
        const auto& data = tensor->data<float>();
        std::cout << name << "(" << tensor->dims() << "): " 
            << std::vector<float>(data, data+tensor->size())
            << std::endl;
    }
    void run() {
        // define workspace
        Workspace workspace;
        //feed blob
        std::vector<float> x(4*3*2);  //define blobs with std::vector
        int count = 0;
	    Tensor value = Tensor(x.size(), caffe2::DeviceType::CPU);
	    for (auto &v: x) {
		v = (float)rand() / RAND_MAX;
		value.mutable_data<float>()[count]=v;
		count++;
	    }
        Blob* my_xBlob = workspace.CreateBlob("X");
		Tensor* tensor = BlobGetMutableTensor(my_xBlob, caffe2::DeviceType::CPU);
        tensor->ResizeLike(value);
        tensor->ShareData(value);

        // create a OperatorDef and run it with workspace
        caffe2::OperatorDef* op_def = new OperatorDef();
        op_def->set_type("Relu");
        op_def->add_input("X");
        print(workspace.GetBlob("X"), "X");
        op_def->add_output("Y");
        // run op
        workspace.RunOperatorOnce(*op_def);
        // print op output
        print(workspace.GetBlob("Y"), "Y");
    }
}

int main(int argc, char** argv) {
    caffe2::GlobalInit(&argc, &argv);
    caffe2::run();
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}