#include <iostream>
#include <exception>
#include <fstream>

#include "glsl_shader.h"

GLSL_shader::GLSL_shader(GLenum shader_type,
                         const char *path):
                            m_shader_type(shader_type)
{
    //getting a pointer to the opengl functions
    QOpenGLFunctions *f = QOpenGLContext::currentContext()->functions();
    //creating the shader
    m_shader_id = f->glCreateShader(m_shader_type);

    //check if allocation worked
    if( 0 == m_shader_id )
    {
        throw std::runtime_error("Error creating vertex shader.");
    }
    //loading the source code
    std::string shaderCode = load_shader_source(path);
    const GLchar* codeArray[] = {shaderCode.c_str()};
    f->glShaderSource( m_shader_id, 1, codeArray, NULL );

    //compile the shader
    f->glCompileShader( m_shader_id );

    //checking compilation result
    GLint result;
    f->glGetShaderiv( m_shader_id, GL_COMPILE_STATUS, &result );
    if( GL_FALSE == result )
    {
        fprintf(stderr, "Vertex shader compilation failed!\n");
        GLint logLen;
        f->glGetShaderiv(m_shader_id, GL_INFO_LOG_LENGTH, &logLen);
        if( logLen > 0 )
        {
            char * log = new char[logLen];
            GLsizei written;
            f->glGetShaderInfoLog(m_shader_id, logLen, &written, log);
            std::cout<<"Shader log:\n"<<log<<std::endl;
            delete [] log;
        }
    }

}

GLSL_shader::~GLSL_shader()
{

}

string GLSL_shader::load_shader_source(const char * path)
{
    // Read the Shader code from the file
    std::string shaderCode;
    std::ifstream shaderStream(path, std::ios::in);
    if(shaderStream.is_open())
    {
        std::string Line = "";
        while(getline(shaderStream, Line))
            shaderCode += "\n" + Line;
        shaderStream.close();
    }
    else
    {
        std::cout<<"welll shit"<<std::endl;
    }

    return shaderCode;
}

GLuint GLSL_shader::get_shader_id() const
{
    return m_shader_id;
}
