import sqlite3

class DataBaseOperations:

    def __init__(self, databasename, tablename):

        self.dbname = databasename
        self.tablename = tablename

    def createDatabase(self):
        try:
            conn = sqlite3.connect(self.dbname)
        except ConnectionError:
            raise ConnectionError
        return conn

    def createTable(self, dictionaryOfcolumnNamesAndcolumnDatatypes):
        try:
            conn = self.createDatabase()
            tableName = self.tablename
            c = conn.cursor()
            for key in dictionaryOfcolumnNamesAndcolumnDatatypes.keys():
                datatype = dictionaryOfcolumnNamesAndcolumnDatatypes[key]
                try:
                    conn.execute(
                        'ALTER TABLE {tableName} ADD COLUMN "{column_name}" {dataType}'.format(tableName=tableName,
                                                                                               column_name=key,
                                                                                               dataType=datatype))
                except:
                    conn.execute('CREATE TABLE {tableName} ({column_name} {dataType})'.format(tableName=tableName,
                                                                                              column_name=key,
                                                                                              dataType=datatype))
            print("Table {0} created in database {1}".format(tableName, self.dbname))
        except Exception as e:

            print("Exception occured: " + str(e))

    def insertIntoTable(self,tablename, listOfvaluesToInsert):
        try:
            conn = self.createDatabase()
            conn.execute('INSERT INTO {tablename}  values ({values})'.format(tablename = tablename,values=(listOfvaluesToInsert)))
            print("Values Inserted Successfully!!!")
            conn.commit()
            #self.closeDbconnection()
        except Exception as e:
            print("Error occured: " + str(e))

    def selectFromTable(self, tablename):

        conn = self.createDatabase()
        c = conn.cursor()
        c.execute("SELECT *  FROM {table}".format(table=tablename))
        print("values in table : " ,c.fetchall())


        # self.closeDbconnection()

    # def deleteTable(self):

    # def updateTable(self,name)

    # def closeDbconnection():


db = DataBaseOperations("testDb","testTable")
db.createDatabase()
tableDetails = {"studentId" : "INTEGER", "studentRoll" : "INTEGER", "studentMarks" : "FLOAT"}
#db.createTable(tableDetails)
valuesToisnert= ('1,1,97')
db.insertIntoTable("testTable",valuesToisnert)
db.selectFromTable("testTable")