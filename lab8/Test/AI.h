#pragma once
#include <iostream>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstdarg>
#include <string>
#include <sstream>
#include <cmath>
#include <fstream>
#include "2048.h"

class experience {
public:
    state sp;
    state spp;
};


class AI {
public:
    static void load_tuple_weights() {
        std::string filename = "0456616.weight";                   // put the name of weight file here
        std::ifstream in;
        in.open(filename.c_str(), std::ios::in | std::ios::binary);
        if (in.is_open()) {
            for (size_t i = 0; i < feature::list().size(); i++) {
                in >> *(feature::list()[i]);
                std::cout << feature::list()[i]->name() << " is loaded from " << filename << std::endl;
            }
            in.close();
        }
    }

    static void set_tuples() {
        feature::list().push_back(new pattern<4>(0, 4, 8, 12));
        feature::list().push_back(new pattern<4>(1, 5, 9, 13));
        feature::list().push_back(new pattern<6>(0, 4, 8, 12, 1, 5));
        feature::list().push_back(new pattern<6>(1, 5, 9, 13, 2, 6));
        //feature::list().push_back(new pattern<4>(4, 5, 8, 9));
        //feature::list().push_back(new pattern<4>(5, 6, 9, 10));
        //feature::list().push_back(new pattern<4>(4, 5, 6, 7));
        //feature::list().push_back(new pattern<6>(4, 5, 6, 7, 10, 11));
    }

    static int get_best_move(state s) {         // return best move dir
        int best_move;
        float max_value = -999;
        for (int opcode = 0; opcode < 4; opcode++) {
            state s_ = s;
            s_.move(opcode);
            int reward = s_.get_reward();
            if (reward == -1)
                continue;
            float value = s_.evaluate_score();
            if (max_value < reward + value) {
                max_value = reward + value;
                best_move = opcode;
            }
        }
        return best_move;
    }

};
