#pragma once
#include <iostream>
#include <glm/glm.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <memory>
class Particle{
    public:
        Particle(float3 pos,float3 velocity, float mass):pos(pos),velocity(velocity),mass(mass){
            force = 0.0f;
            acceleration= make_float3(0.0f,0.0f,0.0f);
        };
        void applyForce(float3 incoming_force, float dt);
        void setVelocity(float3 v);
        void setPosition(float3 p);
        float getMass();
        float getForce();
        float3 getAcceleration();
        float3 getVelocity();
        float3 getPosition();

    private:
        
        float3 acceleration; 
        float3 pos;
        float3 velocity;
        float mass; 
        float force;
};