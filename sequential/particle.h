#pragma once
#include <iostream>
#include <memory>
class Partcile{
    public:
        Partcile(float x, float y):x(x),y(y){};
        std::shared_ptr<float[]> getVertex();
    private:
        float x;
        float y;
        float weight;
};