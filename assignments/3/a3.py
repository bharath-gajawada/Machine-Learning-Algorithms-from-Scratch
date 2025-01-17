import mlp_class1 as mlpc1
import mlp_class2 as mlpc2
import mlp_reg1 as mlpr1
import mlp_reg2 as mlpr2
import autoenc1 as auto1

def run_all():
    mlpc1.main()
    mlpc2.main()
    mlpr1.main()
    mlpr2.main()
    auto1.main()

if __name__ == "__main__":
    run_all()