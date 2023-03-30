#include <leomath/math.h>
#include <iostream>

int main()
{
    leo::Vector4f a(1.0f, 2.0f, 0.0f, 0.0f);
    leo::Vector4f b(2.0f, 1.0f, 0.0f, 0.0f);
    leo::Vector4f c = a + b;
    leo::Vector4f d = a - b;
    leo::Vector4f e = a;
    e *= 2.0f;
    leo::Vector4f f = a / 2.0f;
    auto gt = a <=> f;

    std::cout << leo::FVectorDot(a, b) << std::endl;

    float arr[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    leo::Matrix4x4 mat(arr);
    leo::Matrix4x4 mat2(a, b, c, d);
    leo::Matrix4x4 mat3(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    return 0;
}