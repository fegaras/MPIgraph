/*
 * Copyright © 2025 Leonidas Fegaras, University of Texas at Arlington
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <sstream>
#include <cstdint>
#include <cstring>

using namespace std;

template<typename T>
size_t serialize ( const vector<T> &data, char* &buffer ) {
  ostringstream out;
  size_t len = data.size();
  out.write((const char*)&len,sizeof(size_t));
  for ( T v: data )
    out.write((const char*)&v,sizeof(T));
  size_t size = out.tellp();
  buffer = new char[size];
  memcpy(buffer,out.str().c_str(),size);
  return size;
}

template<typename T>
void deserialize ( vector<T> &data, char* &buffer, const size_t buffer_size ) {
  string s(buffer,buffer_size);
  istringstream in(s);
  size_t old_size = data.size();
  size_t size;
  in.read((char*)&size,sizeof(size_t));
  data.resize(old_size+size);
  for ( int i = 0; i < size; i++ )
    in.read((char*)&data[i+old_size],sizeof(T));
}
