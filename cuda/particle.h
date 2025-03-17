#pragma once
#include <iostream>
#include <glm/glm.hpp>
#include <memory>
class Particle
{
public:
    Particle(float3 pos, float3 velocity, float mass) : pos(pos), velocity(velocity), mass(mass)
    {
        force = 0.0f;
        acceleration = make_float3(0.0f, 0.0f, 0.0f);
    }; 
    Particle() : pos(make_float3(0.0f, 0.0f, 0.0f)),
                 velocity(make_float3(0.0f, 0.0f, 0.0f)),
                 mass(0.0f), force(0.0f), acceleration(make_float3(0.0f, 0.0f, 0.0f)) {}

   __host__ __device__ void applyForce(float3 incoming_force, float dt); 
    void setVelocity(float3 v);                       
    void setPosition(float3 p);                       
   __host__ __device__ float getMass();              
    float getForce();                                
    float3 getAcceleration();                        
    float3 getVelocity();                            
    __host__ __device__ float3 getPosition();                             

private:
    float3 acceleration;
    float3 pos;
    float3 velocity;
    float mass;
    float force;
};