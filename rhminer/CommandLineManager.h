/**
 * Generic command line parser
 *
 * Copyright 2018 Polyminer1 <https://github.com/polyminer1>
 *
 * To the extent possible under law, the author(s) have dedicated all copyright
 * and related and neighboring rights to this software to the public domain
 * worldwide. This software is distributed without any warranty.
 *
 * You should have received a copy of the CC0 Public Domain Dedication along with
 * this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
 */

///
/// @file
/// @copyright Polyminer1, QualiaLibre


#pragma once

#include "corelib/utils.h"
#include <json/reader.h>
using namespace std;

//define a global var as {VAR_NAME} that is registered in the command line manager from pre-main scope
#define RHMINER_COMMAND_LINE_DECLARE_GLOBAL_STRING(OPTION_NAME, VAR_NAME, VAR_CATEGORY, VAR_DESCR) \
    extern string VAR_NAME; \
    template <typename T> struct _CmdLineManagerTrigger##VAR_NAME \
    {  \
        T* _var; \
        _CmdLineManagerTrigger##VAR_NAME(T* v)  \
        {  \
            _var = v; \
            CmdLineManager::GlobalOptions().RegisterValue(OPTION_NAME, VAR_CATEGORY, VAR_DESCR, [&](const string& val){ VAR_NAME = val;}); \
        } \
    }; extern _CmdLineManagerTrigger##VAR_NAME<string> _trigger##VAR_NAME; \

#define RHMINER_COMMAND_LINE_DEFINE_GLOBAL_STRING(VAR_NAME, INIT_VAL)  \
    string VAR_NAME = INIT_VAL; _CmdLineManagerTrigger##VAR_NAME<string> _trigger##VAR_NAME(&VAR_NAME); 


//define a global var as {VAR_NAME} that is registered in the command line manager from pre-main scope
#define RHMINER_COMMAND_LINE_DECLARE_GLOBAL_INT(OPTION_NAME, VAR_NAME, VAR_CATEGORY, VAR_DESCR, VAL_MIN, VAL_MAX) \
    extern int VAR_NAME; \
    template <typename T> struct _CmdLineManagerTrigger##VAR_NAME \
    {  \
        T* _var; \
        _CmdLineManagerTrigger##VAR_NAME(T* v)  \
        {  \
            _var = v; \
            CmdLineManager::GlobalOptions().RegisterValue(OPTION_NAME, VAR_CATEGORY, VAR_DESCR, [&](const string& val){ *_var = RHMINER_ValidateGlobalVarRange(OPTION_NAME, val, VAL_MIN, VAL_MAX);}); \
        } \
    }; extern _CmdLineManagerTrigger##VAR_NAME<int> _trigger##VAR_NAME; \


#define RHMINER_COMMAND_LINE_DEFINE_GLOBAL_INT(VAR_NAME, INIT_VAL)  \
    int VAR_NAME = INIT_VAL; _CmdLineManagerTrigger##VAR_NAME<int> _trigger##VAR_NAME(&VAR_NAME); 



//define a global var as {VAR_NAME} that is registered in the command line manager from pre-main scope
#define RHMINER_COMMAND_LINE_DECLARE_GLOBAL_BOOL(OPTION_NAME, VAR_NAME, VAR_CATEGORY, VAR_DESCR) \
    extern bool VAR_NAME; \
    template <typename T> struct _CmdLineManagerTrigger##VAR_NAME \
    {  \
        T* _var; \
        _CmdLineManagerTrigger##VAR_NAME(T* v)  \
        {  \
            _var = v; \
            CmdLineManager::GlobalOptions().RegisterFlag(OPTION_NAME, VAR_CATEGORY, VAR_DESCR, [&](){ VAR_NAME = true;}); \
        } \
    }; extern _CmdLineManagerTrigger##VAR_NAME<bool> _trigger##VAR_NAME;


#define RHMINER_COMMAND_LINE_DEFINE_GLOBAL_BOOL(VAR_NAME, INIT_VAL) \
    bool VAR_NAME = INIT_VAL; _CmdLineManagerTrigger##VAR_NAME<bool> _trigger##VAR_NAME(&VAR_NAME); 



class CmdLineManager
{
    typedef std::function<void(const string&)> ValFunc;
    typedef std::function<void()> FlagFunc;
    struct CmdLineManagerOption;

public:
    CmdLineManager();
    static CmdLineManager& GlobalOptions();

    void RegisterValue(const string& symbol, const string& cathegory, const string& descriptor, ValFunc f);
    void RegisterValueMultiple(const string& symbol, const string& cathegory, const string& descriptor, ValFunc f);
    void RegisterFlag(const string& symbol, const string& cathegory, const string& descriptor, FlagFunc f);
    void List();
    static void LoadFromXml(const char* configFile);

    bool Parse(int argc, char** argv, bool exitOnError = true);
    bool Parse(const strings& strList , bool exitOnError = true);
    bool PreParseSymbol(const char* symbol);
    void Merge(const CmdLineManager& src);
    bool FindSwitch(const string& switchName);
    void Reset();

    string GetArgsList() {return m_argslist;}
    char** GetArgv() { return m_argv; }
    void   OverrideArgs(const string& newArgs);
    

private:
    static int m_argc;
    static char** m_argv;
    static string m_argslist;
    static Json::Value m_xmlCommandLineConfig;

    struct CmdLineManagerOption
    {
        CmdLineManagerOption() = default;
        CmdLineManagerOption(const CmdLineManagerOption&) = default;

        string cathegory;
        string symbol;
        string descr;
        bool allowMultiples = false;
        bool allowDefault = false;
        ValFunc valSetter;
        FlagFunc flagSetter;
        bool parsed = false;
    };

    CmdLineManagerOption* Find(const string& symb)
    {
        for (int i = 0; i < m_options.size(); i++)
        {
            if (m_options[i].symbol == symb)
                return &m_options[i];
        }
        return NULL;
    }

    void LoadGlobals();
    bool ProcessSymbol(const string& symb, int& i);
    int  ParseInternal(const char* specificSymbol = 0, bool exitOnError = true);
    int  ParseInternalCMD(const char* specificSymbol = 0, bool exitOnError = true);
    int  ParseInternalXML(const char* specificSymbol = 0, bool exitOnError = true);    

    std::vector<CmdLineManagerOption> m_options;
};
extern S32 RHMINER_ValidateGlobalVarRange(const char* varName, const string& valStr, S32 _min, S32 _max);

